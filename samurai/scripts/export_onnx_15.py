import argparse
import math
import os
import sys

import numpy as np
import onnx
import torch
import torch.nn as nn
import torch.nn.functional as F
from onnxsim import simplify

# -----------------------------------------------------------------------------
# Monkey Patching for Opset 15 and TensorRT 8.5.2 Compatibility
# -----------------------------------------------------------------------------


def monkey_patch_sam2():
    print("Applying monkey patches for Opset 15 and TensorRT 8.5.2 compatibility...")

    # 1. Patch Scaled Dot Product Attention (SDPA)
    # Opset 15 exporter has trouble decomposing the native SDPA.
    def manual_sdpa(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None):
        if scale is None:
            scale = q.shape[-1] ** -0.5

        # Matrix multiplication for attention scores
        attn = (q @ k.transpose(-2, -1)) * scale

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
                attn_mask = new_mask
            attn = attn + attn_mask

        attn = torch.softmax(attn, dim=-1)

        if dropout_p > 0.0:
            attn = F.dropout(attn, p=dropout_p)

        return attn @ v

    F.scaled_dot_product_attention = manual_sdpa

    # 2. Patch interpolate (antialias is not supported in Opset 15 Resize)
    _old_interpolate = F.interpolate

    def patched_interpolate(
        input,
        size=None,
        scale_factor=None,
        mode="nearest",
        align_corners=None,
        recompute_scale_factor=None,
        antialias=False,
    ):
        # Force antialias=False for ONNX export compatibility
        return _old_interpolate(
            input,
            size=size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
            recompute_scale_factor=recompute_scale_factor,
            antialias=False,
        )

    F.interpolate = patched_interpolate

    # 3. Patch RoPEAttention.forward to avoid in-place slice assignment
    # ONNX exporter often fails on k[:, :, :num_k_rope] = ...
    sys.path.append("./sam2")
    from sam2.modeling.position_encoding import get_rotation_matrices
    from sam2.modeling.sam.transformer import (
        USE_MAT_ROTARY_ENC,
        RoPEAttention,
        apply_rotary_enc,
        apply_rotary_matenc,
    )

    def patched_rope_forward(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_k_exclude_rope: int = 0
    ) -> torch.Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Apply rotary position encoding
        w = h = math.sqrt(q.shape[-2])
        self.freqs_cis = self.freqs_cis.to(q.device)
        if self.freqs_cis.shape[0] != q.shape[-2]:
            self.freqs_cis = self.compute_cis(end_x=int(w), end_y=int(h)).to(q.device)

        if USE_MAT_ROTARY_ENC:
            if self.rotmats.shape[2] != q.shape[-2]:
                self.rotmats = get_rotation_matrices(
                    dim=self.internal_dim // self.num_heads,
                    end_x=int(w),
                    end_y=int(h),
                    theta=self.rope_theta,
                    device=q.device,
                    dtype=q.dtype,
                )
            self.rotmats = self.rotmats.to(q.device)

        if q.shape[-2] != k.shape[-2]:
            assert self.rope_k_repeat

        num_k_rope = k.size(-2) - num_k_exclude_rope

        if USE_MAT_ROTARY_ENC:
            q_out, k_rope = apply_rotary_matenc(
                q,
                k[:, :, :num_k_rope],
                rotmats=self.rotmats,
                repeat_freqs_k=self.rope_k_repeat,
            )
        else:
            q_out, k_rope = apply_rotary_enc(
                q,
                k[:, :, :num_k_rope],
                freqs_cis=self.freqs_cis,
                repeat_freqs_k=self.rope_k_repeat,
            )

        # FIX: Avoid in-place assignment to avoid ONNX export errors
        if num_k_exclude_rope > 0:
            k_out = torch.cat([k_rope, k[:, :, num_k_rope:]], dim=-2)
        else:
            k_out = k_rope

        dropout_p = self.dropout_p if self.training else 0.0
        # Attention (will use the patched manual_sdpa above)
        out = F.scaled_dot_product_attention(q_out, k_out, v, dropout_p=dropout_p)

        out = self._recombine_heads(out)
        out = self.out_proj(out)

        return out

    RoPEAttention.forward = patched_rope_forward
    print("Monkey patches applied successfully.")


# Apply patches before other sam2 imports to ensure they take effect
monkey_patch_sam2()

# -----------------------------------------------------------------------------
# SAM2 Export Logic
# -----------------------------------------------------------------------------

from sam2.build_sam import build_sam2_video_predictor
from sam2.modeling.sam2_base import NO_OBJ_SCORE, SAM2Base, get_1d_sine_pe

# 不使用科学计数法
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)


class ImageEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base):
        super().__init__()
        self.model = sam_model

    def forward(self, image):
        """Run the forward pass on the given image."""

        # get the image features
        img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[None, :, None, None]
        img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[None, :, None, None]
        image = image.permute(0, 3, 1, 2).float() / 255.0
        image -= img_mean
        image /= img_std
        backbone_out = self.model.forward_image(image)
        # expand the features to have the same dimension as the number of objects
        batch_size = 1
        expanded_image = image.expand(batch_size, -1, -1, -1)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            expanded_backbone_out["backbone_fpn"][i] = feat.expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self.model._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features

        _, _, current_vision_feats, current_vision_pos_embeds, feat_sizes = features

        # current_vision_feats [65536, 1, 32], [16384, 1, 64], [4096, 1, 256]
        # (HW)BC => BCHW
        high_res_features = [
            x.permute(1, 2, 0).view(x.size(1), x.size(2), *s)
            for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
        ]
        # for 1st frame, the memory is empty, so we add a no-memory embedding
        B = batch_size
        C = self.model.hidden_dim  # 256
        H, W = feat_sizes[-1]  # 64, 64
        pix_feat_with_mem = (
            current_vision_feats[-1] + self.model.no_mem_embed
        )  # (4096, 1, 256) + (1, 1, 256)
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(
            B, C, H, W
        )  # (4096, 1, 256) -> (1, 256, 4096) -> (1, 256, 64, 64)

        return (
            *high_res_features,
            current_vision_feats[-1],
            current_vision_pos_embeds[-1],
            pix_feat_with_mem,
        )


class MemoryAttention(nn.Module):
    def __init__(self, sam_model: SAM2Base):
        super().__init__()
        self.model = sam_model

    def forward(
        self,
        current_vision_feats,
        current_vision_pos_embeds,
        maskmem_feats,
        maskmem_pos_enc,
        obj_ptrs,
        obj_pos,
    ):
        B = current_vision_feats.size(1)
        C = self.model.hidden_dim
        # 动态计算 H, W
        feat_len = current_vision_feats.shape[0]
        H = W = int(math.sqrt(feat_len))

        maskmem_feats = maskmem_feats.flatten(0, 1)
        maskmem_pos_enc = maskmem_pos_enc.flatten(0, 1)

        obj_ptrs = obj_ptrs.reshape(
            -1, B, C // self.model.mem_dim, self.model.mem_dim
        )
        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)
        memory = torch.cat([maskmem_feats, obj_ptrs], dim=0)

        t_diff_max = self.model.max_obj_ptrs_in_encoder - 1
        tpos_dim = C
        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
        obj_pos = self.model.obj_ptr_tpos_proj(obj_pos)
        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.model.mem_dim)
        obj_pos = obj_pos.repeat_interleave(C // self.model.mem_dim, dim=0)

        memory_pos_embed = torch.cat([maskmem_pos_enc, obj_pos], dim=0)

        pix_feat_with_mem = self.model.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=obj_ptrs.size(0),
        )

        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)

        return pix_feat_with_mem


class MemoryEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base):
        super().__init__()
        self.model = sam_model

    def forward(
        self, current_vision_feat, pred_masks_high_res, object_score_logits, is_mask_from_pts
    ):
        B = current_vision_feat.size(1)
        C = self.model.hidden_dim
        # 动态计算 H, W
        feat_len = current_vision_feat.shape[0]
        H = W = int(math.sqrt(feat_len))

        pix_feat = current_vision_feat.permute(1, 2, 0).view(B, C, H, W)

        binarize = self.model.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts

        mask_for_mem0 = (pred_masks_high_res > 0).float()
        mask_for_mem1 = torch.sigmoid(pred_masks_high_res)
        mask_for_mem = torch.where(binarize, mask_for_mem0, mask_for_mem1)

        mask_for_mem = mask_for_mem * self.model.sigmoid_scale_for_mem_enc
        mask_for_mem = mask_for_mem + self.model.sigmoid_bias_for_mem_enc

        maskmem_out = self.model.memory_encoder(
            pix_feat,
            mask_for_mem,
            skip_mask_sigmoid=True,
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]

        if self.model.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.model.no_obj_embed_spatial[..., None, None].expand(*maskmem_features.shape)

        maskmem_features = maskmem_features.flatten(2).permute(2, 0, 1)
        maskmem_pos_enc = maskmem_out["vision_pos_enc"][0].flatten(2).permute(2, 0, 1)

        return maskmem_features, maskmem_pos_enc


class MaskDecoder(nn.Module):
    def __init__(self, sam_model: SAM2Base):
        super().__init__()
        self.model = sam_model

    def forward(
        self, point_coords, point_labels, pix_feat, high_res_features_0, high_res_features_1
    ):
        high_res_features = [high_res_features_0, high_res_features_1]

        sparse_embeddings = self.model.sam_prompt_encoder._embed_points(
            point_coords, point_labels, pad=True
        )

        # 动态匹配尺寸
        feat_h, feat_w = pix_feat.shape[-2:]
        dense_embeddings = self.model.sam_prompt_encoder.no_mask_embed.weight.reshape(
            1, -1, 1, 1
        ).expand(
            1, -1, feat_h, feat_w
        )

        # 动态生成 PE
        image_pe = self.model.sam_prompt_encoder.pe_layer((feat_h, feat_w)).unsqueeze(0)

        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.model.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,
            high_res_features=high_res_features,
        )

        is_obj_appearing = object_score_logits > self.model.min_obj_score_logits

        low_res_multimasks = torch.where(
            is_obj_appearing[:, None, None],
            low_res_multimasks,
            NO_OBJ_SCORE,
        )

        obj_ptr = self.model.obj_ptr_proj(sam_output_tokens)

        lambda_is_obj_appearing = is_obj_appearing.float()
        obj_ptr = lambda_is_obj_appearing * obj_ptr
        obj_ptr = obj_ptr + (1 - lambda_is_obj_appearing) * self.model.no_obj_ptr.repeat(
            1, obj_ptr.size(1), 1
        )

        return low_res_multimasks, ious, obj_ptr, object_score_logits, self.model.maskmem_tpos_enc


def export_sam2_onnx(sam2_model, config):
    sam2_model.eval()
    os.makedirs(config.onnx_path, exist_ok=True)

    # 1. Export image_encoder
    image_encoder = ImageEncoder(sam2_model)
    export_sam2_image_encoder(image_encoder, config)

    # 2. Export memory_attention
    memory_attention = MemoryAttention(sam2_model)
    export_sam2_memory_attention(memory_attention, config)

    # 3. Export memory_encoder
    memory_encoder = MemoryEncoder(sam2_model)
    export_sam2_memory_encoder(memory_encoder, config)

    # 4. Export mask decoder
    mask_decoder = MaskDecoder(sam2_model)
    export_sam2_mask_decoder(mask_decoder, config)

    print(f"\033[93mExported sam2 model to {config.onnx_path} (Opset 15)\033[0m")


def export_sam2_image_encoder(image_encoder, config):
    dummy_input = torch.ones(1, 512, 512, 3)
    onnx_path = f"{config.onnx_path}/image_encoder.onnx"

    torch.onnx.export(
        image_encoder,
        dummy_input,
        onnx_path,
        verbose=False,
        input_names=["image"],
        output_names=[
            "high_res_features0",
            "high_res_features1",
            "low_res_features",
            "vision_pos_embeds",
            "pix_feat_with_mem",
        ],
        opset_version=15,
    )

    if config.simplify_onnx:
        simplify_model, check = simplify(onnx_path)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save_model(simplify_model, onnx_path.replace(".onnx", "_simplified.onnx"))


def export_sam2_memory_attention(memory_attention, config):
    current_vision_feats = torch.ones(1024, 1, 256)
    current_vision_pos_embeds = torch.ones(1024, 1, 256)
    n, m = 7, 16
    maskmem_feats = torch.ones(1024, n, 1, 64)
    memory_pos_embed = torch.ones(1024, n, 1, 64)
    obj_ptrs = torch.ones(m, 1, 256)
    obj_pos = torch.arange(m) + 1
    obj_pos = torch.cat([obj_pos[-1:], obj_pos[:-1]]).to(dtype=torch.int32)

    dummy_input = (
        current_vision_feats,
        current_vision_pos_embeds,
        maskmem_feats,
        memory_pos_embed,
        obj_ptrs,
        obj_pos,
    )

    onnx_path = f"{config.onnx_path}/memory_attention.onnx"
    torch.onnx.export(
        memory_attention,
        dummy_input,
        onnx_path,
        verbose=False,
        input_names=[
            "current_vision_feats",
            "current_vision_pos_embeds",
            "maskmem_feats",
            "memory_pos_embed",
            "obj_ptrs",
            "obj_pos",
        ],
        output_names=["pix_feat_with_mem"],
        opset_version=15,
        dynamic_axes={
            "maskmem_feats": {1: "num_feat"},
            "memory_pos_embed": {1: "num_pos_enc"},
            "obj_ptrs": {0: "num_obj_ptr"},
            "obj_pos": {0: "num_obj_pos"},
        },
    )

    if config.simplify_onnx:
        simplify_model, check = simplify(onnx_path)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save_model(simplify_model, onnx_path.replace(".onnx", "_simplified.onnx"))


def export_sam2_memory_encoder(memory_encoder, config):
    pix_feat = torch.ones(1024, 1, 256)
    high_res_mask_for_mem = torch.ones(1, 1, 512, 512)
    object_score_logits = torch.ones([1, 1])
    is_mask_from_pts = torch.tensor(0, dtype=torch.bool)

    dummy_input = (pix_feat, high_res_mask_for_mem, object_score_logits, is_mask_from_pts)
    onnx_path = f"{config.onnx_path}/memory_encoder.onnx"

    torch.onnx.export(
        memory_encoder,
        dummy_input,
        onnx_path,
        verbose=False,
        input_names=["pix_feat", "mask_for_mem", "object_score_logits", "is_mask_from_pts"],
        output_names=["maskmem_features", "maskmem_pos_enc"],
        opset_version=15,
    )

    if config.simplify_onnx:
        simplify_model, check = simplify(onnx_path)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save_model(simplify_model, onnx_path.replace(".onnx", "_simplified.onnx"))


def export_sam2_mask_decoder(mask_decoder, config):
    pix_feat_with_mem = torch.ones([1, 256, 32, 32])
    high_res_features_0 = torch.ones([1, 32, 128, 128])
    high_res_features_1 = torch.ones([1, 64, 64, 64])
    point_coords = torch.ones([1, 2, 2])
    point_labels = torch.ones([1, 2], dtype=torch.int32)

    dummy_input = (
        point_coords,
        point_labels,
        pix_feat_with_mem,
        high_res_features_0,
        high_res_features_1,
    )

    onnx_path = f"{config.onnx_path}/mask_decoder.onnx"
    torch.onnx.export(
        mask_decoder,
        dummy_input,
        onnx_path,
        verbose=False,
        input_names=[
            "point_coords",
            "point_labels",
            "pix_feat_with_mem",
            "high_res_features_0",
            "high_res_features_1",
        ],
        output_names=[
            "low_res_multimasks",
            "ious",
            "obj_ptr",
            "object_score_logits",
            "maskmem_tpos_enc",
        ],
        opset_version=15,
    )

    if config.simplify_onnx:
        simplify_model, check = simplify(onnx_path)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save_model(simplify_model, onnx_path.replace(".onnx", "_simplified.onnx"))


def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")


def main(args):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cpu")

    with torch.inference_mode():
        export_sam2_onnx(predictor, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="./sam2/checkpoints/sam2.1_hiera_small.pt")
    parser.add_argument("--use_fp16", action="store_true")
    parser.add_argument("--simplify_onnx", action="store_true")
    parser.add_argument("--onnx_path", default="./onnx_model_opset15")
    args = parser.parse_args()
    main(args)
