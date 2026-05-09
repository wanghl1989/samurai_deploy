from collections import OrderedDict

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


def determine_model_cfg(model_path):
    """Determine model config based on checkpoint path."""
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


def process_image(
    image,
    image_size,
    offload_video_to_cpu,
    compute_device="cuda",
):
    temp_img = cv2.resize(image, (image_size, image_size))
    temp_img = torch.from_numpy(temp_img).permute(2, 0, 1).float() / 255.0

    img_mean = torch.tensor(IMAGE_MEAN, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(IMAGE_STD, dtype=torch.float32)[:, None, None]
    print(img_mean.shape, temp_img.shape)
    if not offload_video_to_cpu:
        temp_img = temp_img.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)

    temp_img -= img_mean
    temp_img /= img_std
    return temp_img


class SamuraiTracker:
    def __init__(self, model_path, device="cuda", use_fp16=False) -> None:
        model_cfg = determine_model_cfg(model_path)
        self.predictor = build_sam2_video_predictor(model_cfg, model_path, device=str(device))

        self.use_fp16 = True
        self.state = None
    
    def reset(self):
        self.state = None

    def init_video(self, *_args, **_kwargs): ...

    def add_first_frame_bbox(self, image, bbox):
        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() and self.use_fp16 else torch.float32,
        ):
            self.state = self.init_state(image)
            self.predictor.add_new_points_or_box(self.state, frame_idx=0, obj_id=0, box=bbox)
            self.predictor.propagate_in_video_preflight(self.state)
            return np.zeros(image.shape[:2], dtype=np.bool)

    def add_first_frame_points(self, image, points):

        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() and self.use_fp16 else torch.float32,
        ):
            self.state = self.init_state(image)
            labels = [1 for _ in range(len(points))]
            self.predictor.add_new_points_or_box(
                self.state, points=points, labels=labels, frame_idx=0, obj_id=0
            )
            self.predictor.propagate_in_video_preflight(self.state)
            return np.zeros(image.shape[:2], dtype=np.bool)

    def init_state(
        self,
        first_image,
        offload_video_to_cpu=True,
        offload_state_to_cpu=True,
    ):

        compute_device = self.predictor.device
        video_height, video_width = first_image.shape[:2]
        first_frame = process_image(
            first_image,
            self.predictor.image_size,
            offload_video_to_cpu,
            compute_device=compute_device,
        )
        inference_state = {}
        inference_state["images"] = [first_frame]
        inference_state["num_frames"] = 1
        # whether to offload the video frames to CPU memory
        # turning on this option saves the GPU memory with only a very small overhead
        inference_state["offload_video_to_cpu"] = offload_video_to_cpu
        # whether to offload the inference state to CPU memory
        # turning on this option saves the GPU memory at the cost of a lower tracking fps
        # (e.g. in a test case of 768x768 model, fps dropped from 27 to 24 when tracking one object
        # and from 24 to 21 when tracking two objects)
        inference_state["offload_state_to_cpu"] = offload_state_to_cpu
        # the original video height and width, used for resizing final output scores
        inference_state["video_height"] = video_height
        inference_state["video_width"] = video_width
        inference_state["device"] = compute_device
        if offload_state_to_cpu:
            inference_state["storage_device"] = torch.device("cpu")
        else:
            inference_state["storage_device"] = compute_device
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        self.predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
        # Warm up the visual backbone and cache the image feature on frame 0
        return inference_state

    def track_step(self, image):
        # output_dict = self.state["output_dict"]
        # obj_ids = self.state["obj_ids"]
        masks = self.process_video_frames(self.state, image)
        return masks[0][0].cpu().numpy()

    def process_video_frames(self, inference_state, frame):

        with (
            torch.inference_mode(),
            torch.autocast(
                device_type="cuda" if torch.cuda.is_available() else "cpu",
                dtype=torch.float16
                if torch.cuda.is_available() and self.use_fp16
                else torch.float32,
            ),
        ):
            output_dict = inference_state["output_dict"]
            batch_size = self.predictor._get_obj_num(inference_state)
            frame_idx = inference_state["num_frames"]
            frame_torch = process_image(
                frame,
                self.predictor.image_size,
                inference_state["offload_video_to_cpu"],
                compute_device=inference_state["device"],
            )

            inference_state["images"].append(frame_torch)
            inference_state["num_frames"] += 1

            reverse = False

            storage_key = "non_cond_frame_outputs"
            current_out, pred_masks = self.predictor._run_single_frame_inference(
                inference_state=inference_state,
                output_dict=output_dict,
                frame_idx=frame_idx,
                batch_size=batch_size,
                is_init_cond_frame=False,
                point_inputs=None,
                mask_inputs=None,
                reverse=reverse,
                run_mem_encoder=True,
            )
            output_dict[storage_key][frame_idx] = current_out
            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self.predictor._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self.predictor._get_orig_video_res_output(
                inference_state, pred_masks
            )

            return video_res_masks
