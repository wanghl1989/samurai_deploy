import os
import onnxruntime as ort
import numpy as np
import argparse
import time

import cv2
# import decord
import torch
from tqdm import tqdm

from kalman_filter import KalmanFilter

color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]

# 打印onnxruntime的provider
providers = ort.get_all_providers()
print("onnxruntime provider: ", providers)

if 'CUDAExecutionProvider' in providers:
    print("CUDAExecutionProvider is available.")
    provider = ['CUDAExecutionProvider']
else:
    print("CUDAExecutionProvider is not available, using CPUExecutionProvider instead.")
    provider = ['CPUExecutionProvider']

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        x, y, w, h = map(float, line.split(','))
        x, y, w, h = int(x), int(y), int(w), int(h)
        prompts[fid] = ((x, y, x + w, y + h), 0)
    return prompts


class CameraPredictor:
    def __init__(self, args):
        self.model_path = args.model_path
        providers = ort.get_available_providers()
        self.provider = "CUDAExecutionProvider" if "CUDAExecutionProvider" in providers else "CPUExecutionProvider"
        self.use_fp16 = args.use_fp16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.init_models()

        self.img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1)).astype(np.float32)
        self.img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1)).astype(np.float32)

        self.image_size = 512
        # self.video_W, self.video_H = 1280, 720
        
        self.maskmem_tpos_enc = None

        self.memory_bank = {}
        self.kf = KalmanFilter()
        self.kf_mean = None
        self.kf_covariance = None
        self.stable_frames = 0

        self.stable_frames_threshold = 15
        self.stable_ious_threshold = 0.3
        self.kf_score_weight = 0.25
        self.memory_bank_iou_threshold = 0.5
        self.memory_bank_obj_score_threshold = 0.0
        self.memory_bank_kf_score_threshold = 0.0
        self.max_obj_ptrs_in_encoder = 16
        self.num_maskmem = 7

    def init_models(self):
        if not self.use_fp16:
            self.image_encoder_session = ort.InferenceSession(args.model_path + "/image_encoder.onnx", providers=provider)
            self.memory_attention_session = ort.InferenceSession(args.model_path + "/memory_attention.onnx", providers=provider)
            self.memory_encoder_session = ort.InferenceSession(args.model_path + "/memory_encoder.onnx", providers=provider)
            self.mask_decoder_session = ort.InferenceSession(args.model_path + "/mask_decoder.onnx", providers=provider)
        else:
            self.image_encoder_session = ort.InferenceSession(args.model_path + "/image_encoder_FP16.onnx", providers=provider)
            self.memory_attention_session = ort.InferenceSession(args.model_path + "/memory_attention_FP16.onnx", providers=provider)
            self.memory_encoder_session = ort.InferenceSession(args.model_path + "/memory_encoder_FP16.onnx", providers=provider)
            self.mask_decoder_session = ort.InferenceSession(args.model_path + "/mask_decoder_FP16.onnx", providers=provider)

        self.image_encoder_input = self.image_encoder_session.get_inputs()
        self.image_encoder_output = self.image_encoder_session.get_outputs()

        self.memory_attention_input = self.memory_attention_session.get_inputs()
        self.memory_attention_output = self.memory_attention_session.get_outputs()

        self.memory_encoder_input = self.memory_encoder_session.get_inputs()
        self.memory_encoder_output = self.memory_encoder_session.get_outputs()

        self.mask_decoder_input = self.mask_decoder_session.get_inputs()
        self.mask_decoder_output = self.mask_decoder_session.get_outputs()

        self.image_encoder_input_names = [inputs.name for inputs in self.image_encoder_input]
        self.image_encoder_output_names = [outputs.name for outputs in self.image_encoder_output]

        self.memory_attention_input_names = [inputs.name for inputs in self.memory_attention_input]
        self.memory_attention_output_names = [outputs.name for outputs in self.memory_attention_output]

        self.memory_encoder_input_names = [inputs.name for inputs in self.memory_encoder_input]
        self.memory_encoder_output_names = [outputs.name for outputs in self.memory_encoder_output]

        self.mask_decoder_input_names = [inputs.name for inputs in self.mask_decoder_input]
        self.mask_decoder_output_names = [outputs.name for outputs in self.mask_decoder_output]

    def add_first_frame_bbox(self, frame_idx, image, first_frame_bbox):
        '''
        Add the first bbox when the frame_idx is 0.
        '''
        video_H, video_W = image.shape[:2]
        input_image = cv2.resize(image, (self.image_size, self.image_size))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

        input_image = input_image.astype(np.float32)
        # image_encoder predict
        image_encoder_outputs = self.image_encoder_session.run(self.image_encoder_output_names, 
                                                          {self.image_encoder_input_names[0]: input_image[np.newaxis, ...]})
        high_res_features0, high_res_features1, low_res_features, _, pix_feat_with_mem = image_encoder_outputs

        box_coords = np.array(first_frame_bbox).reshape((1, 2, 2))
        box_labels = np.array([2, 3]).reshape((1, 2))

        points = box_coords / np.array([video_W, video_H])

        input_points = (points * self.image_size).astype(np.float32)
        input_labels = box_labels.astype(np.int32)

        mask_decoder_outputs = self.mask_decoder_session.run(self.mask_decoder_output_names, {
                                            self.mask_decoder_input_names[0]: input_points,
                                            self.mask_decoder_input_names[1]: input_labels,
                                            self.mask_decoder_input_names[2]: pix_feat_with_mem,
                                            self.mask_decoder_input_names[3]: high_res_features0,
                                            self.mask_decoder_input_names[4]: high_res_features1,
                                        })
        _, ious, obj_ptrs, object_score_logits, maskmem_tpos_enc = mask_decoder_outputs

        self.maskmem_tpos_enc = maskmem_tpos_enc

        pred_mask, high_res_masks_for_mem, best_iou_inds, kf_score = self._forward_sam_head(mask_decoder_outputs)

        # memory_encoder predict
        is_mask_from_pts = np.array([frame_idx==0]).astype(bool)
        memory_encoder_outputs = self.memory_encoder_session.run(self.memory_encoder_output_names, {
                                            self.memory_encoder_input_names[0]: low_res_features,
                                            self.memory_encoder_input_names[1]: high_res_masks_for_mem,
                                            self.memory_encoder_input_names[2]: object_score_logits,
                                            self.memory_encoder_input_names[3]: is_mask_from_pts,
                                        })
        maskmem_features, maskmem_pos_enc = memory_encoder_outputs

        self.memory_bank[0] = {
                                'maskmem_features': maskmem_features,
                                'maskmem_pos_enc': maskmem_pos_enc,
                                'obj_ptr': obj_ptrs[0, best_iou_inds],
                                'best_iou_score': ious[0, best_iou_inds],
                                'obj_score_logits': object_score_logits,
                                'kf_score': kf_score,
                                }
        
        return pred_mask[0][0]

    def track_step(self, frame_idx, image):
        # print(f"\033[93mframe_idx: {frame_idx}\033[0m")

        # step 1:image_encoder predict, get image feature
        start = time.time()
        input_image = cv2.resize(image, (self.image_size, self.image_size))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.astype(np.float32)

        image_encoder_outputs = self.image_encoder_session.run(self.image_encoder_output_names, 
                                                          {self.image_encoder_input_names[0]: input_image[np.newaxis, ...]})
        # print("image_encoder inference time: ", (time.time() - start) * 1000, "ms")
        high_res_features0, high_res_features1, low_res_features, vision_pos_embeds, _ = image_encoder_outputs

        # step 2: memory_attention predict
        input_points = np.zeros((1, 2, 2), dtype=np.float32)
        input_labels = -np.ones((1, 2), dtype=np.int32)
    
        memmask_features = [self.memory_bank[0]['maskmem_features'].copy()]
        memmask_pos_enc = [self.memory_bank[0]['maskmem_pos_enc'] + self.maskmem_tpos_enc[6]]
        object_ptrs = [self.memory_bank[0]['obj_ptr'].copy()]
        ## samurai----------------------------------------------- ##
        valid_indices = [] 
        if frame_idx > 1:
            for i in range(frame_idx - 1, 0, -1):  # Iterate backwards through previous frames
                iou_score = self.memory_bank[i]["best_iou_score"]  # Get mask affinity score
                obj_score = self.memory_bank[i]["obj_score_logits"]  # Get object score
                kf_score = self.memory_bank[i]["kf_score"]  # Get motion score if available
                # Check if the scores meet the criteria for being a valid index
                if iou_score > self.memory_bank_iou_threshold and \
                    obj_score > self.memory_bank_obj_score_threshold and \
                    (kf_score is None or kf_score > self.memory_bank_kf_score_threshold):
                    valid_indices.insert(0, i)
                # Check the number of valid indices
                if len(valid_indices) >= self.max_obj_ptrs_in_encoder - 1:
                    break
            # valid_indices.insert(0, 1)

        # print("valid_indices: ", valid_indices, 'prev_frame_idx : ')
        # 最近6帧的memmask_features
        for prev_frame_idx in valid_indices[::-1]:
            # print(prev_frame_idx, end=', ')
            mem = self.memory_bank.get(prev_frame_idx, None)
            if mem is not None:
                memmask_features.insert(1, mem['maskmem_features'].copy())
                memmask_pos_enc.insert(1, mem['maskmem_pos_enc'].copy())
            if len(memmask_features) >= self.num_maskmem:
                break
        # print()
        ## samurai----------------------------------------------- ##

        obj_pos_enc = np.arange(1, frame_idx)[:15]
        obj_pos_enc = np.insert(obj_pos_enc, 0, frame_idx).astype(np.int32)
        # print('obj_pos_enc : ', obj_pos_enc)
        # 最近15帧的object_ptrs
        # print("object_ptrs: ")
        for i in range(frame_idx - 15, frame_idx):
            if i < 1:
                continue
            mem = self.memory_bank.get(i, None)
            if mem is not None:
                # print(i, end=', ')
                object_ptrs.append(mem['obj_ptr'].copy())
        # print()

        for i, pos_enc in enumerate(reversed(memmask_pos_enc[1:])):
            pos_enc[:] = pos_enc[:] + self.maskmem_tpos_enc[i]
        
        memory = np.concatenate(memmask_features, axis=0)
        memory_pos_embed = np.concatenate(memmask_pos_enc, axis=0)
        memory = memory.reshape(-1, len(memmask_features), memory.shape[-2], memory.shape[-1])
        memory_pos_embed = memory_pos_embed.reshape(-1, len(memmask_pos_enc), memory_pos_embed.shape[-2], memory_pos_embed.shape[-1])

        object_ptrs = object_ptrs[0:1] + object_ptrs[1:][::-1]
        object_ptrs = np.stack(object_ptrs, axis=0)

        start = time.time()
        memory_attention_outputs = self.memory_attention_session.run(self.memory_attention_output_names, {
                                            self.memory_attention_input_names[0]: low_res_features,
                                            self.memory_attention_input_names[1]: vision_pos_embeds,
                                            self.memory_attention_input_names[2]: memory,
                                            self.memory_attention_input_names[3]: memory_pos_embed,
                                            self.memory_attention_input_names[4]: object_ptrs,
                                            self.memory_attention_input_names[5]: obj_pos_enc,
                                            # self.memory_attention_input_names[6]: is_init_cond_frame,
                                        })
        # print("memory_attention inference time: ", (time.time() - start) * 1000, "ms")
        pix_feat_with_mem = memory_attention_outputs[0]

        # step 3 : mask decoder predict
        start = time.time()
        mask_decoder_outputs = self.mask_decoder_session.run(self.mask_decoder_output_names, {
                                            self.mask_decoder_input_names[0]: input_points,
                                            self.mask_decoder_input_names[1]: input_labels,
                                            self.mask_decoder_input_names[2]: pix_feat_with_mem,
                                            self.mask_decoder_input_names[3]: high_res_features0,
                                            self.mask_decoder_input_names[4]: high_res_features1,
                                            # self.mask_decoder_input_names[5]: np.array([self.video_W]).astype(np.int32),
                                            # self.mask_decoder_input_names[6]: np.array([self.video_H]).astype(np.int32),
                                        })
        # print("mask_decoder inference time: ", (time.time() - start) * 1000, "ms")
        _, ious, obj_ptrs, object_score_logits, maskmem_tpos_enc = mask_decoder_outputs
        self.maskmem_tpos_enc = maskmem_tpos_enc

        pred_mask, high_res_masks_for_mem, best_iou_inds, kf_score = self._forward_sam_head(mask_decoder_outputs)

        # step 4 : memory_encoder predict, save maskmem to memory bank
        is_mask_from_pts = np.array([frame_idx==0]).astype(bool)
        start = time.time()
        memory_encoder_outputs = self.memory_encoder_session.run(self.memory_encoder_output_names, {
                                            self.memory_encoder_input_names[0]: low_res_features,
                                            self.memory_encoder_input_names[1]: high_res_masks_for_mem,
                                            self.memory_encoder_input_names[2]: object_score_logits,
                                            self.memory_encoder_input_names[3]: is_mask_from_pts,
                                        })
        # print("memory_encoder inference time: ", (time.time() - start) * 1000, "ms")
        maskmem_features, maskmem_pos_enc = memory_encoder_outputs

        self.memory_bank[frame_idx] = {
                                'maskmem_features': maskmem_features,
                                'maskmem_pos_enc': maskmem_pos_enc,
                                'obj_ptr': obj_ptrs[0, best_iou_inds],
                                'best_iou_score': ious[0, best_iou_inds],
                                'obj_score_logits': object_score_logits,
                                'kf_score': kf_score,
                                }
        
        return pred_mask[0][0]

    def _normalize_image(self, image):
        if isinstance(image, np.ndarray):
            image = image.transpose((2, 0, 1)).astype(np.float32)
            image /= 255.0
            image -= self.img_mean
            image /= self.img_std
        elif isinstance(image, torch.Tensor):
            # image = image.to(self.device)
            image = image.permute(2, 0, 1).float()
            image /= 255.0
            image -= self.img_mean
            image /= self.img_std
            image = image.cpu().numpy()

        return image

    def _forward_sam_head(self, mask_decoder_outputs):
        low_res_multimasks, ious, _, _, _ = mask_decoder_outputs
        # high_res_multimasks = F.interpolate(low_res_multimasks, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        high_res_multimasks = cv2.resize(low_res_multimasks[0].transpose(1, 2, 0), (self.image_size, self.image_size))
        high_res_multimasks = high_res_multimasks.transpose(2, 0, 1)[None, ...]

        ## samurai ---------------------------------------------------------------------##
        B = 1
        kf_ious = None
        if self.kf_mean is None and self.kf_covariance is None or self.stable_frames == 0:
            best_iou_inds = np.argmax(ious, axis=-1)
            batch_inds = np.arange(B)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds][:, None]
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds][:, None]
            non_zero_indices = np.argwhere(high_res_masks[0][0] > 0.0)
            if len(non_zero_indices) == 0:
                high_res_bbox = [0, 0, 0, 0]
            else:
                y_min, x_min = non_zero_indices.min(axis=0).tolist()
                y_max, x_max = non_zero_indices.max(axis=0).tolist()
                high_res_bbox = [x_min, y_min, x_max, y_max]
            self.kf_mean, self.kf_covariance = self.kf.initiate(self.kf.xyxy_to_xyah(high_res_bbox))

            self.stable_frames += 1
        elif self.stable_frames < self.stable_frames_threshold:
            self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
            best_iou_inds = np.argmax(ious, axis=-1)
            batch_inds = np.arange(B)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds][:, None]
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds][:, None]
            non_zero_indices = np.argwhere(high_res_masks[0][0] > 0.0)
            if len(non_zero_indices) == 0:
                high_res_bbox = [0, 0, 0, 0]
            else:
                y_min, x_min = non_zero_indices.min(axis=0).tolist()
                y_max, x_max = non_zero_indices.max(axis=0).tolist()
                high_res_bbox = [x_min, y_min, x_max, y_max]
            if ious[0][best_iou_inds] > self.stable_ious_threshold:
                self.kf_mean, self.kf_covariance = self.kf.update(self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(high_res_bbox))
                self.stable_frames += 1
            else:
                self.stable_frames = 0
        else:
            self.kf_mean, self.kf_covariance = self.kf.predict(self.kf_mean, self.kf_covariance)
            high_res_multibboxes = []
            batch_inds = np.arange(B)
            for i in range(ious.shape[1]):
                non_zero_indices = np.argwhere(high_res_multimasks[batch_inds, i][0] > 0.0)
                if len(non_zero_indices) == 0:
                    high_res_multibboxes.append([0, 0, 0, 0])
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    high_res_multibboxes.append([x_min, y_min, x_max, y_max])
            # compute the IoU between the predicted bbox and the high_res_multibboxes
            kf_ious = np.array(self.kf.compute_iou(self.kf_mean[:4], high_res_multibboxes))
            # weighted iou
            weighted_ious = self.kf_score_weight * kf_ious + (1 - self.kf_score_weight) * ious
            best_iou_inds = np.argmax(weighted_ious, axis=-1)
            batch_inds = torch.arange(B)
            low_res_masks = low_res_multimasks[batch_inds, best_iou_inds][:, None]
            high_res_masks = high_res_multimasks[batch_inds, best_iou_inds][:, None]

            if ious[0][best_iou_inds] < self.stable_ious_threshold:
                self.stable_frames = 0
            else:
                self.kf_mean, self.kf_covariance = self.kf.update(self.kf_mean, self.kf_covariance, self.kf.xyxy_to_xyah(high_res_multibboxes[best_iou_inds.item()]))
        
        best_iou_score = ious[0][best_iou_inds]
        kf_score = kf_ious[best_iou_inds] if kf_ious is not None else None
        pred_mask = low_res_masks
        high_res_masks_for_mem = high_res_masks

        return pred_mask, high_res_masks_for_mem, best_iou_inds, kf_score


def main(args):
    camera_predictor = CameraPredictor(args)

    image_size = camera_predictor.image_size
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error: Could not open video : ", args.video_path)
        return
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # decord.bridge.set_bridge("torch")
    # vr = decord.VideoReader(args.video_path, width=image_size, height=image_size)
    # vr = decord.VideoReader(args.video_path)

    if args.save_to_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('./result.mp4', fourcc, 30, (frame_width, frame_height))

    name_window = os.path.basename(args.video_path)
    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)

    start = time.time()
    for frame_idx in tqdm(range(num_frames), desc="Processing video frames"):
        # start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx == 0:
            bbox = cv2.selectROI(name_window, frame.copy())
            print("bbox (x, y, w, h): ", bbox)
            x, y, w, h = bbox
            first_frame_bbox = (x, y, x + w, y + h)
            # first_frame_bbox = load_txt(args.txt_path)[0][0]

            mask = camera_predictor.add_first_frame_bbox(frame_idx, frame.copy(), first_frame_bbox)
        else:
            mask = camera_predictor.track_step(frame_idx, frame.copy())

        mask = cv2.resize(mask, (frame_width, frame_height))
        mask = mask > 0.0
        non_zero_indices = np.argwhere(mask)
        if len(non_zero_indices) == 0:
            bbox = [0, 0, 0, 0]
        else:
            y_min, x_min = non_zero_indices.min(axis=0).tolist()
            y_max, x_max = non_zero_indices.max(axis=0).tolist()
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        mask_img = np.zeros((frame_height, frame_width, 3), np.uint8)
        mask_img[mask] = color[1]

        frame = cv2.addWeighted(frame, 1, mask_img, 0.2, 0)
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), color[1], 2)
        cv2.imshow(name_window, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if args.save_to_video:
            out.write(frame)
        # print("Time taken: ", (time.time() - start) * 1000, "ms")

    elapsed = (time.time() - start) * 1000
    print(f"spend time: {elapsed:.2f}ms")
    print(f"every frame spend time: {elapsed / (frame_idx + 1):.2f}ms")
    print(f"fps: {1000 / (elapsed / (frame_idx + 1)):.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", default="../assets/1917.mp4", help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", default="first_frame_bbox.txt", help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="../onnx_model", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", default=True, help="Save results to a video.")
    parser.add_argument("--use_fp16", default=True, help="Use FP16 precision for inference.")
    args = parser.parse_args()

    main(args)