import argparse
import os
import sys

import numpy as np
import onnx
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

# from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
from onnxruntime.transformers.float16 import convert_float_to_float16
from onnxsim import simplify
from torch import nn

sys.path.append("./sam2")
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

        # current_vision_feats[-1].permute(1, 2, 0).view(B, C, H, W)

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
    ):  # , is_init_cond_frame):
        B = current_vision_feats.size(1)
        C = self.model.hidden_dim
        # H, W = (64, 64) # feat_sizes[-1] imagesize=1024
        H, W = (32, 32)  # feat_sizes[-1] imagesize=512
        # H, W = (16, 16) # feat_sizes[-1] imagesize=256

        # # for 1st frame
        # pix_feat_with_mem1 = current_vision_feats + self.model.no_mem_embed

        # for after 1st frame
        # maskmem_feats前两维展开，后两维保持不变
        maskmem_feats = maskmem_feats.flatten(0, 1)  # (4096, n, 1, 64) -> (4096*n, 1, 64)
        maskmem_pos_enc = maskmem_pos_enc.flatten(0, 1)  # (4096, n, 1, 64) -> (4096*n, 1, 64)

        obj_ptrs = obj_ptrs.reshape(
            -1, B, C // self.model.mem_dim, self.model.mem_dim
        )  # (n, 1, 256) -> (n, 1, 4, 64)
        obj_ptrs = obj_ptrs.permute(0, 2, 1, 3).flatten(0, 1)  # (n, 4, 1, 64) -> (4n, 1, 64)
        memory = torch.cat([maskmem_feats, obj_ptrs], dim=0)

        t_diff_max = self.model.max_obj_ptrs_in_encoder - 1
        tpos_dim = C
        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
        obj_pos = self.model.obj_ptr_tpos_proj(obj_pos)
        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.model.mem_dim)
        obj_pos = obj_pos.repeat_interleave(C // self.model.mem_dim, dim=0)

        # maskmem_pos_enc += self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1] # 放在forward外面处理
        memory_pos_embed = torch.cat([maskmem_pos_enc, obj_pos], dim=0)

        pix_feat_with_mem = self.model.memory_attention(
            curr=current_vision_feats,
            curr_pos=current_vision_pos_embeds,
            memory=memory,
            memory_pos=memory_pos_embed,
            num_obj_ptr_tokens=obj_ptrs.size(0),
        )

        # pix_feat_with_mem = torch.where(is_init_cond_frame, pix_feat_with_mem1, pix_feat_with_mem2)

        # reshape the output to (HW)BC => BCHW
        pix_feat_with_mem = pix_feat_with_mem.permute(1, 2, 0).view(B, C, H, W)

        return pix_feat_with_mem


class MemoryEncoder(nn.Module):
    def __init__(self, sam_model: SAM2Base):
        super().__init__()
        self.model = sam_model

    def forward(
        self, current_vision_feat, pred_masks_high_res, object_score_logits, is_mask_from_pts
    ):
        """
        The memory encoder generates a memory by downsampling the output mask using
        a convolutional module and summing it element-wise with the unconditioned frame
        embedding from the
        image-encoder
        """

        B = current_vision_feat.size(1)  # batch size on this frame, 1
        C = self.model.hidden_dim  #
        # H, W = (64, 64) # H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size # imagesize=1024
        H, W = (
            32,
            32,
        )  # H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size # imagesize=512
        # H, W = (16, 16) # H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size # imagesize=256
        # top-level feature, (HW)BC => BCHW
        pix_feat = current_vision_feat.permute(1, 2, 0).view(B, C, H, W)

        # scale the raw mask logits with a temperature before applying sigmoid
        binarize = self.model.binarize_mask_from_pts_for_mem_enc and is_mask_from_pts

        mask_for_mem0 = (pred_masks_high_res > 0).float()

        # apply sigmoid on the raw mask logits to turn them into range (0, 1)
        mask_for_mem1 = torch.sigmoid(pred_masks_high_res)

        mask_for_mem = torch.where(binarize, mask_for_mem0, mask_for_mem1)

        # apply scale and bias terms to the sigmoid probabilities
        mask_for_mem = mask_for_mem * self.model.sigmoid_scale_for_mem_enc

        mask_for_mem = mask_for_mem + self.model.sigmoid_bias_for_mem_enc

        maskmem_out = self.model.memory_encoder(
            pix_feat,
            mask_for_mem,
            skip_mask_sigmoid=True,  # sigmoid already applied
        )
        maskmem_features = maskmem_out["vision_features"]
        maskmem_pos_enc = maskmem_out["vision_pos_enc"]
        # add a no-object embedding to the spatial memory to indicate that the frame
        # is predicted to be occluded (i.e. no object is appearing in the frame)
        if self.model.no_obj_embed_spatial is not None:
            is_obj_appearing = (object_score_logits > 0).float()
            maskmem_features += (
                1 - is_obj_appearing[..., None, None]
            ) * self.model.no_obj_embed_spatial[..., None, None].expand(*maskmem_features.shape)

        maskmem_features = maskmem_features.flatten(2).permute(
            2, 0, 1
        )  # (1, 64, 64, 64) -> (1, 64, 4096) -> (4096, 1, 64)
        maskmem_pos_enc = (
            maskmem_pos_enc[0].flatten(2).permute(2, 0, 1)
        )  # (1, 64, 64, 64) -> (1, 64, 4096) -> (4096, 1, 64)

        return maskmem_features, maskmem_pos_enc


class MaskDecoder(nn.Module):
    def __init__(self, sam_model: SAM2Base):
        super().__init__()
        self.model = sam_model

    def forward(
        self, point_coords, point_labels, pix_feat, high_res_features_0, high_res_features_1
    ):  # , video_W, video_H):
        """Forward SAM prompt encoders and mask heads."""
        # point_inputs = {'point_coords': point_coords,
        #                 'point_labels': point_labels}
        # multimask_output = True # is False when add_new_points_or_box
        high_res_features = [high_res_features_0, high_res_features_1]

        # sam_outputs = self.model._forward_sam_heads(
        #                                             backbone_features=pix_feat,
        #                                             point_inputs=point_inputs,
        #                                             mask_inputs=None,
        #                                             high_res_features=high_res_features,
        #                                             multimask_output=multimask_output,
        #                                         )
        # (
        #     low_res_multimasks,
        #     high_res_multimasks,
        #     ious,
        #     low_res_masks,
        #     high_res_masks_for_mem,
        #     obj_ptr,
        #     object_score_logits,
        #     best_iou_score,
        #     kf_ious
        # ) = sam_outputs
        # return low_res_multimasks, high_res_multimasks, ious, obj_ptr, object_score_logits, self.model.maskmem_tpos_enc

        # sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
        #                                             points=(point_coords, point_labels),
        #                                             boxes=None,
        #                                             masks=None,
        #                                         )
        sparse_embeddings = self.model.sam_prompt_encoder._embed_points(
            point_coords, point_labels, pad=True
        )
        dense_embeddings = self.model.sam_prompt_encoder.no_mask_embed.weight.reshape(
            1, -1, 1, 1
        ).expand(
            1,
            -1,
            self.model.sam_prompt_encoder.image_embedding_size[0],
            self.model.sam_prompt_encoder.image_embedding_size[1],
        )

        (
            low_res_multimasks,
            ious,
            sam_output_tokens,
            object_score_logits,
        ) = self.model.sam_mask_decoder(
            image_embeddings=pix_feat,
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=True,
            repeat_image=False,  # the image is already batched
            high_res_features=high_res_features,
        )

        is_obj_appearing = object_score_logits > self.model.min_obj_score_logits

        # Mask used for spatial memories is always a *hard* choice between obj and no obj,
        # consistent with the actual mask prediction
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


def print_model_info(model):
    print("==== Model ====")
    print(model)

    # print("==== Model Parameters ====")
    # # print parameters
    # for name, param in model.named_parameters():
    #     print(name, ' : ',  param.size())
    # compute the number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", num_params / 1e6, "M")


def export_sam2_onnx(sam2_model, config):
    """
    #导出sam2模型为onnx格式

    sam2包括四个模型,
    分别是image_encoder, memory_attention, memory_encoder, prompt_encoder with mask decoder
    """
    sam2_model.eval()

    os.makedirs(config.onnx_path, exist_ok=True)

    # 1. Export image_encoder
    image_encoder = ImageEncoder(sam2_model)
    export_sam2_image_encoder(image_encoder, config)  # onnx_path=f"{onnx_path}/image_encoder.onnx")

    # 2. Export memory_attention
    memory_attention = MemoryAttention(sam2_model)
    export_sam2_memory_attention(
        memory_attention, config
    )  # onnx_path=f"{onnx_path}/memory_attention.onnx")

    # 3. Export memory_encoder
    memory_encoder = MemoryEncoder(sam2_model)
    export_sam2_memory_encoder(
        memory_encoder, config
    )  # onnx_path=f"{onnx_path}/memory_encoder.onnx")

    # 4. Export mask decoder
    mask_decoder = MaskDecoder(sam2_model)
    export_sam2_mask_decoder(mask_decoder, config)  # onnx_path=f"{onnx_path}/mask_decoder.onnx")

    print(f"\033[93mExported sam2 model to {config.onnx_path}\033[0m")


def export_sam2_image_encoder(image_encoder: ImageEncoder, config):
    """
    #导出sam2的image_encoder模型为onnx格式
    """
    # print model info
    # print_model_info(image_encoder)

    # dummy_input = torch.ones(1, *input_shape) # [1, 3, 1024, 1024]
    # dummy_input = torch.ones(1, 1024, 1024, 3)
    dummy_input = torch.ones(1, 512, 512, 3)
    # dummy_input = torch.ones(1, 256, 256, 3)
    torch_outputs = image_encoder(dummy_input)
    (
        high_res_features0,
        high_res_features1,
        low_res_features,
        vision_pos_embeds,
        pix_feat_with_mem,
    ) = torch_outputs
    print(
        "high_res_features0: ",
        high_res_features0.size(),
        high_res_features0.sum(),
        type(high_res_features0),
    )  # [1, 32, 256, 256] [1, 32, 128, 128]
    print(
        "high_res_features1: ", high_res_features1.size(), high_res_features1.sum()
    )  # [1, 64, 128, 128] [1, 64, 64, 64]
    print(
        "low_res_features: ", low_res_features.size(), low_res_features.sum()
    )  # [4096, 1, 256] [1024, 1, 256]
    print(
        "vision_pos_embeds: ", vision_pos_embeds.size(), vision_pos_embeds.sum()
    )  # [4096, 1, 256] [1024, 1, 256]
    print(
        "pix_feat_with_mem: ", pix_feat_with_mem.size(), pix_feat_with_mem.sum()
    )  # [4096, 1, 256] [1024, 1, 256]

    # print('image_encoder FLOAT32 FLOPS: ', FlopCountAnalysis(image_encoder, dummy_input).total()/1e9, "GFLOPs")
    # print('image_encoder Parameters: ', parameter_count_table(image_encoder))

    # 保存vision_pos_embeds为文件用于ascend推理
    if config.save_constants:
        os.makedirs("./om_model", exist_ok=True)
        vision_pos_embeds.detach().numpy().astype(np.float32).tofile(
            "./om_model/vision_pos_embeds.bin"
        )
        np.save("./om_model/vision_pos_embeds.npy", vision_pos_embeds.detach().numpy())

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
        opset_version=17,
        # dynamic_axes={
        #     "image": {0 : "height", 1: "width"},
        # }
    )

    # check onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    # onnx.save_model(SymbolicShapeInference.infer_shapes(onnx_model, auto_merge=True), onnx_path)  # type: ignore

    if config.use_fp16:
        print(
            "Converting to FP16. Warning: This FP16 model should NOT be used for TensorRT. TRT provides its own fp16 option."
        )
        onnx.save_model(
            convert_float_to_float16(onnx_model, keep_io_types=True),
            onnx_path.replace(".onnx", "_FP16.onnx"),
        )

        # feed_dict = {'image': dummy_input.numpy()}
        # model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(onnx_model, feed_dict, rtol=0.1, atol=0.1, keep_io_types=True)
        # onnx.save(model_fp16, onnx_path.replace(".onnx", "_FP16.onnx"))

    if config.quantize_onnx:  # 只导出INT8模型，没有测试过推理
        quantize_dynamic(
            model_input=onnx_path,
            model_output=onnx_path.replace(".onnx", "_INT8.onnx"),
            # per_channel=True,  # Set to True if per-channel quantization is desired
            op_types_to_quantize=["Conv", "MatMul"],  # Op types to quantize
            weight_type=QuantType.QUInt8,  # Specify the weight type for quantization
        )

    if config.simplify_onnx:
        # simplify onnx model
        simplify_model, check = simplify(onnx_path)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save_model(simplify_model, onnx_path.replace(".onnx", "_simplified.onnx"))

    print(f"\033[93mExported sam2 image_encoder model to {onnx_path}\033[0m")


def export_sam2_memory_attention(memory_attention: MemoryAttention, config):
    """
    #导出sam2的memory_attention模型为onnx格式
    """
    # print_model_info(memory_attention)

    # current_vision_feats = torch.ones(4096, 1, 256)  #[torch.Size([4096, 1, 256])]
    # current_vision_pos_embeds = torch.ones(4096, 1, 256)
    current_vision_feats = torch.ones(1024, 1, 256)  # [torch.Size([4096, 1, 256])]
    current_vision_pos_embeds = torch.ones(1024, 1, 256)
    # current_vision_feats = torch.ones(256, 1, 256)  #[torch.Size([4096, 1, 256])]
    # current_vision_pos_embeds = torch.ones(256, 1, 256)

    n = 7
    m = 16
    # maskmem_feats = torch.ones(4096, n, 1, 64)
    # memory_pos_embed = torch.ones(4096, n, 1, 64)
    maskmem_feats = torch.ones(1024, n, 1, 64)
    memory_pos_embed = torch.ones(1024, n, 1, 64)
    # maskmem_feats = torch.ones(256, n, 1, 64)
    # memory_pos_embed = torch.ones(256, n, 1, 64)
    obj_ptrs = torch.ones(m, 1, 256)
    obj_pos = torch.arange(m) + 1
    obj_pos = torch.cat([obj_pos[-1:], obj_pos[:-1]]).to(
        dtype=torch.int32
    )  # [16, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

    # is_init_cond_frame = torch.tensor(0, dtype=torch.bool)

    dummy_input = (
        current_vision_feats,
        current_vision_pos_embeds,
        maskmem_feats,
        memory_pos_embed,
        obj_ptrs,
        obj_pos,
    )

    pix_feat_with_mem = memory_attention(*dummy_input)
    print(
        "pix_feat_with_mem : ",
        pix_feat_with_mem.size(),
        pix_feat_with_mem.sum(),
        pix_feat_with_mem.min(),
        pix_feat_with_mem.max(),
        pix_feat_with_mem.mean(),
        pix_feat_with_mem.std(),
    )
    # output pix_feat_with_mem  torch.Size([4096, 1, 256])

    # print('memory_attention FLOAT32 FLOPS: ', FlopCountAnalysis(memory_attention, dummy_input).total()/1e9, "GFLOPs")

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
        opset_version=17,
        dynamic_axes={
            # "current_vision_feats": {0: "num_feat"},
            # "current_vision_pos_embeds": {0: "num_pos_enc"},
            "maskmem_feats": {1: "num_feat"},
            "memory_pos_embed": {1: "num_pos_enc"},
            "obj_ptrs": {0: "num_obj_ptr"},
            "obj_pos": {0: "num_obj_pos"},
        },
    )

    # check onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    # onnx.save_model(SymbolicShapeInference.infer_shapes(onnx_model, auto_merge=True), onnx_path)  # type: ignore

    if config.use_fp16:
        print(
            "Converting to FP16. Warning: This FP16 model should NOT be used for TensorRT. TRT provides its own fp16 option."
        )
        onnx.save_model(
            convert_float_to_float16(onnx_model, keep_io_types=True),
            onnx_path.replace(".onnx", "_FP16.onnx"),
        )
        # feed_dict = {'current_vision_feats': current_vision_feats.numpy(),
        #              'current_vision_pos_embeds': current_vision_pos_embeds.numpy(),
        #              'maskmem_feats': maskmem_feats.numpy(),
        #              'memory_pos_embed': memory_pos_embed.numpy(),
        #              'obj_ptrs': obj_ptrs.numpy(),
        #              'obj_pos': obj_pos.numpy()
        #             }
        # model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(onnx_model, feed_dict, rtol=0.1, atol=0.1, keep_io_types=True)
        # onnx.save(model_fp16, onnx_path.replace(".onnx", "_FP16.onnx"))

    if config.quantize_onnx:
        quantize_dynamic(
            model_input=onnx_path,
            model_output=onnx_path.replace(".onnx", "_INT8.onnx"),
            # per_channel=True,  # Set to True if per-channel quantization is desired
            op_types_to_quantize=["Conv", "MatMul"],  # Op types to quantize
            weight_type=QuantType.QUInt8,  # Specify the weight type for quantization
        )

    if config.simplify_onnx:
        # simplify onnx model
        simplify_model, check = simplify(onnx_path)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save_model(simplify_model, onnx_path.replace(".onnx", "_simplified.onnx"))
    print(f"\033[93mExported sam2 memory_attention model to {onnx_path}\033[0m")


def export_sam2_memory_encoder(memory_encoder: MemoryEncoder, config):
    """
    #导出sam2的memory_encoder模型为onnx格式
    """
    # print_model_info(memory_encoder)

    # pix_feat = torch.ones(4096, 1, 256)
    pix_feat = torch.ones(1024, 1, 256)
    # pix_feat = torch.ones(256, 1, 256)
    # high_res_mask_for_mem = torch.ones(1, 1, 1024, 1024)
    high_res_mask_for_mem = torch.ones(1, 1, 512, 512)
    # high_res_mask_for_mem = torch.ones(1, 1, 256, 256)
    object_score_logits = torch.ones([1, 1])
    is_mask_from_pts = torch.tensor(
        0, dtype=torch.bool
    )  # for the first frame is True, for the rest is False
    # skip_mask_sigmoid=True

    dummy_input = (pix_feat, high_res_mask_for_mem, object_score_logits, is_mask_from_pts)
    torch_outputs = memory_encoder(*dummy_input)
    maskmem_features, maskmem_pos_enc = torch_outputs
    print("maskmem_features: ", maskmem_features.size(), maskmem_features.sum())
    print("maskmem_pos_enc: ", maskmem_pos_enc.size(), maskmem_pos_enc.sum())

    # print('memory_encoder FLOAT32 FLOPS: ', FlopCountAnalysis(memory_encoder, dummy_input).total()/1e9, "GFLOPs")

    # 保存maskmem_pos_enc为文件用于ascend推理
    if config.save_constants:
        os.makedirs("./om_model", exist_ok=True)
        maskmem_pos_enc.detach().numpy().astype(np.float32).tofile("./om_model/maskmem_pos_enc.bin")
        np.save("./om_model/maskmem_pos_enc.npy", maskmem_pos_enc.detach().numpy())

    onnx_path = f"{config.onnx_path}/memory_encoder.onnx"
    torch.onnx.export(
        memory_encoder,
        dummy_input,
        onnx_path,
        verbose=False,
        input_names=["pix_feat", "mask_for_mem", "object_score_logits", "is_mask_from_pts"],
        output_names=["maskmem_features", "maskmem_pos_enc"],
        opset_version=17,
        # dynamic_axes = {
        #                 "pix_feat": {0: "num_feat"},
        #                 "mask_for_mem": {2: "height", 3: "width"},
        #             }
    )

    # check onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    # onnx.save_model(SymbolicShapeInference.infer_shapes(onnx_model, auto_merge=True), onnx_path)  # type: ignore

    if config.use_fp16:
        print(
            "Converting to FP16. Warning: This FP16 model should NOT be used for TensorRT. TRT provides its own fp16 option."
        )
        onnx.save_model(
            convert_float_to_float16(onnx_model, keep_io_types=True),
            onnx_path.replace(".onnx", "_FP16.onnx"),
        )
        # feed_dict = {'pix_feat': pix_feat.numpy(),
        #              'mask_for_mem': high_res_mask_for_mem.numpy(),
        #              'object_score_logits': object_score_logits.numpy(),
        #              'is_mask_from_pts': is_mask_from_pts.numpy()
        #             }
        # model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(onnx_model, feed_dict, rtol=0.1, atol=0.1, keep_io_types=True)
        # onnx.save(model_fp16, onnx_path.replace(".onnx", "_FP16.onnx"))

    if config.quantize_onnx:
        quantize_dynamic(
            model_input=onnx_path,
            model_output=onnx_path.replace(".onnx", "_INT8.onnx"),
            # per_channel=True,  # Set to True if per-channel quantization is desired
            op_types_to_quantize=["Conv", "MatMul"],  # Op types to quantize
            weight_type=QuantType.QUInt8,  # Specify the weight type for quantization
        )

    if config.simplify_onnx:
        # simplify onnx model
        simplify_model, check = simplify(onnx_path)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save_model(simplify_model, onnx_path.replace(".onnx", "_simplified.onnx"))
    print(f"\033[93mExported sam2 memory_encoder model to {onnx_path}\033[0m")


def export_sam2_mask_decoder(mask_decoder: MaskDecoder, config):
    """
    #导出sam2的mask_decoder模型为onnx格式
    """
    # print_model_info(mask_decoder)

    # pix_feat_with_mem = torch.ones([1, 256, 64, 64])
    pix_feat_with_mem = torch.ones([1, 256, 32, 32])
    # pix_feat_with_mem = torch.ones([1, 256, 16, 16])

    # high_res_features_0 = torch.ones([1, 32, 256, 256])
    # high_res_features_1 = torch.ones([1, 64, 128, 128])
    high_res_features_0 = torch.ones([1, 32, 128, 128])
    high_res_features_1 = torch.ones([1, 64, 64, 64])
    # high_res_features_0 = torch.ones([1, 32, 64, 64])
    # high_res_features_1 = torch.ones([1, 64, 32, 32])

    # point_coords = torch.tensor([[[307.2000, 432.3556],
    #                             [580.8000, 881.7778]]], dtype=torch.float)
    # point_labels = torch.tensor([[2, 3]], dtype=torch.int32)
    # point_labels = torch.tensor([[2., 3.]])
    # video_W = torch.tensor([1280], dtype=torch.int32)
    # video_H = torch.tensor([720], dtype=torch.int32)

    # point_coords = torch.ones([1, 3, 256])
    # point_labels = torch.ones([1, 256, 32, 32])
    point_coords = torch.ones([1, 2, 2])
    point_labels = torch.ones([1, 2], dtype=torch.int32)

    dummy_input = (
        point_coords,
        point_labels,
        pix_feat_with_mem,
        high_res_features_0,
        high_res_features_1,
        #    video_W, video_H
    )

    torch_outputs = mask_decoder(*dummy_input)
    for i, o in enumerate(torch_outputs):
        print(f" torch output_{i}: ", o.size(), o.sum(), o if o.numel() <= 3 else None)

    low_res_multimasks, ious, obj_ptrs, object_score_logits, maskmem_tpos_enc = torch_outputs

    # print('mask_decoder FLOAT32 FLOPS: ', FlopCountAnalysis(mask_decoder, dummy_input).total()/1e9, "GFLOPs")

    # 保存 maskmem_tpos_enc for ascend推理
    if config.save_constants:
        os.makedirs("./om_model", exist_ok=True)
        maskmem_tpos_enc.detach().numpy().astype(np.float32).tofile(
            "./om_model/maskmem_tpos_enc.bin"
        )
        np.save("./om_model/maskmem_tpos_enc.npy", maskmem_tpos_enc.detach().numpy())

    # low_res_multimasks, ious, sam_output_tokens, object_score_logits = torch_outputs
    # print("low_res_multimasks: ", low_res_multimasks.size()) # torch.Size([1, 3, 256, 256])
    # print("ious: ", ious.size()) # torch.Size([1, 3])
    # print("sam_output_tokens: ", sam_output_tokens.size()) # torch.Size([1, 3, 256])
    # print("object_score_logits: ", object_score_logits.size()) # torch.Size([1, 1])

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
            #  "video_W", "video_H"
        ],
        # output_names=["pred_masks",
        #               "high_res_masks_for_mem",
        #               "object_score_logits",
        #               "obj_ptr",
        #               "ious",
        #               "maskmem_tpos_enc"],
        output_names=[
            "low_res_multimasks",
            #   "high_res_multimasks",
            "ious",
            "obj_ptr",
            "object_score_logits",
            "maskmem_tpos_enc",
        ],
        opset_version=17,
        # dynamic_axes={
        # "point_coords": {1: "num_points"},
        # "point_labels": {1: "num_points"},
        # "pix_feat_with_mem": {2: "height", 3: "width"},
        # "high_res_features_0": {2: "height", 3: "width"},
        # "high_res_features_1": {2: "height", 3: "width"},
        # "video_W": {0: "video_W"},
        # "video_H": {0: "video_H"},
        # }
    )

    # check onnx model
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    # onnx.save_model(SymbolicShapeInference.infer_shapes(onnx_model, auto_merge=True), onnx_path)

    if config.use_fp16:
        print(
            "Converting to FP16. Warning: This FP16 model should NOT be used for TensorRT. TRT provides its own fp16 option."
        )
        onnx.save_model(
            convert_float_to_float16(onnx_model, keep_io_types=True),
            onnx_path.replace(".onnx", "_FP16.onnx"),
        )
        # feed_dict = {'point_coords': point_coords.numpy(),
        #              'point_labels': point_labels.numpy(),
        #              'pix_feat_with_mem': pix_feat_with_mem.numpy(),
        #              'high_res_features_0': high_res_features_0.numpy(),
        #              'high_res_features_1': high_res_features_1.numpy(),
        #             #  'video_W': video_W.numpy(),
        #             #  'video_H': video_H.numpy()
        #             }
        # model_fp16 = auto_mixed_precision.auto_convert_mixed_precision(onnx_model, feed_dict, rtol=0.1, atol=0.1, keep_io_types=True)
        # onnx.save(model_fp16, onnx_path.replace(".onnx", "_FP16.onnx"))

    if config.quantize_onnx:
        quantize_dynamic(
            model_input=onnx_path,
            model_output=onnx_path.replace(".onnx", "_INT8.onnx"),
            # per_channel=True,  # Set to True if per-channel quantization is desired
            op_types_to_quantize=["Conv", "MatMul"],  # Op types to quantize
            weight_type=QuantType.QUInt8,  # Specify the weight type for quantization
        )

    if config.simplify_onnx:
        # simplify onnx model
        simplify_model, check = simplify(onnx_path)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save_model(simplify_model, onnx_path.replace(".onnx", "_simplified.onnx"))
    print(f"\033[93mExported sam2 mask_decoder model to {onnx_path}\033[0m")


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
    # print_model_info(predictor)

    with torch.inference_mode():
        export_sam2_onnx(predictor, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        default="./sam2/checkpoints/sam2.1_hiera_tiny.pt",
        help="Path to the model checkpoint.",
    )
    parser.add_argument("--use_fp16", default=False, help="export fp16 onnx model.")
    parser.add_argument("--simplify_onnx", default=False, help="Simplify onnx model.")
    parser.add_argument("--quantize_onnx", default=False, help="Quantize onnx model to int8.")
    parser.add_argument(
        "--save_constants",
        default=False,
        help="Save constant tensors to .bin files for Ascend inference.",
    )
    parser.add_argument("--onnx_path", default="./onnx_model", help="Path to save the ONNX model.")
    args = parser.parse_args()
    main(args)

