#!/usr/bin/env python3
"""
Interactive Real-time Tracking Script
Click on the first frame to select a target, then track it through the video.
"""

import argparse
import gc
import os.path as osp
from collections import OrderedDict

import cv2
import numpy as np
import torch
from sam2.build_sam import build_sam2_video_predictor

color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255)]


class PointSelector:
    """Handle mouse clicks to select points on the first frame."""

    def __init__(self, window_name, image):
        self.window_name = window_name
        self.image = image.copy()
        self.points = []
        self.labels = []

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        # Display instructions
        display_img = self.image.copy()
        h, w = display_img.shape[:2]
        cv2.putText(
            display_img,
            "Click to select target (left click: foreground, right click: background)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_img,
            "Press SPACE to start tracking, ESC to exit",
            (10, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow(window_name, display_img)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click: foreground point (label=1)
            self.points.append([x, y])
            self.labels.append(1)
            self._update_display()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click: background point (label=0)
            self.points.append([x, y])
            self.labels.append(0)
            self._update_display()

    def _update_display(self):
        """Update the display with clicked points."""
        display_img = self.image.copy()

        # Draw clicked points
        for point, label in zip(self.points, self.labels):
            color = (0, 255, 0) if label == 1 else (0, 0, 255)
            cv2.circle(display_img, tuple(point), 5, color, -1)
            cv2.circle(display_img, tuple(point), 7, (255, 255, 255), 2)

        # Redraw instructions
        cv2.putText(
            display_img,
            f"Points: {len(self.points)} (Press SPACE to start, ESC to exit)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_img,
            "Left click: foreground | Right click: background",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

        cv2.imshow(self.window_name, display_img)

    def wait_for_selection(self):
        """Wait for user to complete selection and press SPACE."""
        print("\n=== Point Selection Mode ===")
        print("- Left click: Add foreground point (green)")
        print("- Right click: Add background point (red)")
        print("- Press SPACE: Start tracking")
        print("- Press ESC: Exit")
        print()

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                return None, None
            elif key == 32:  # SPACE
                if len(self.points) > 0:
                    break
                else:
                    print("Please select at least one point!")

        cv2.destroyWindow(self.window_name)

        # Convert to numpy arrays
        points = np.array(self.points, dtype=np.float32)
        labels = np.array(self.labels, dtype=np.int32)

        print(
            f"Selected {len(points)} points ({np.sum(labels == 1)} foreground, {np.sum(labels == 0)} background)"
        )

        return points, labels


def process_image(
    image,
    image_size,
    offload_video_to_cpu,
    img_mean=(0.485, 0.456, 0.406),
    img_std=(0.229, 0.224, 0.225),
    compute_device="cuda",
):
    temp_img = cv2.resize(image, (image_size, image_size))
    temp_img = torch.from_numpy(temp_img).permute(2, 0, 1).float() / 255.0

    img_mean = torch.tensor(img_mean, dtype=torch.float32)[:, None, None]
    img_std = torch.tensor(img_std, dtype=torch.float32)[:, None, None]
    if not offload_video_to_cpu:
        temp_img = temp_img.to(compute_device)
        img_mean = img_mean.to(compute_device)
        img_std = img_std.to(compute_device)

    temp_img -= img_mean
    temp_img /= img_std
    return temp_img


def init_state(predictor, offload_video_to_cpu, offload_state_to_cpu, first_image):
    compute_device = predictor.device
    video_height, video_width = first_image.shape[:2]
    first_frame = process_image(
        first_image, predictor.image_size, offload_video_to_cpu, compute_device=compute_device
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
    predictor._get_image_feature(inference_state, frame_idx=0, batch_size=1)
    # Warm up the visual backbone and cache the image feature on frame 0
    return inference_state


def process_video_frames(predictor, inference_state, frame):
    output_dict = inference_state["output_dict"]
    batch_size = predictor._get_obj_num(inference_state)
    frame_idx = inference_state["num_frames"]
    frame_torch = process_image(
        frame,
        predictor.image_size,
        inference_state["offload_video_to_cpu"],
        compute_device=inference_state["device"],
    )

    inference_state["images"].append(frame_torch)
    inference_state["num_frames"] += 1

    reverse = False

    storage_key = "non_cond_frame_outputs"
    current_out, pred_masks = predictor._run_single_frame_inference(
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
    predictor._add_output_per_object(inference_state, frame_idx, current_out, storage_key)
    inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

    # Resize the output mask to the original video resolution (we directly use
    # the mask scores on GPU for output to avoid any CPU conversion in between)
    _, video_res_masks = predictor._get_orig_video_res_output(inference_state, pred_masks)

    return video_res_masks


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


def mask_to_bbox(mask):
    """Convert binary mask to bounding box [x, y, w, h]."""
    non_zero_indices = np.argwhere(mask)
    if len(non_zero_indices) == 0:
        return [0, 0, 0, 0]
    else:
        y_min, x_min = non_zero_indices.min(axis=0).tolist()
        y_max, x_max = non_zero_indices.max(axis=0).tolist()
        return [x_min, y_min, x_max - x_min, y_max - y_min]


def main(args):
    print("=" * 20)
    print("SAMURAI Interactive Real-time Tracking")
    print("=" * 20)

    # Check if video exists
    if not osp.exists(args.video_path):
        print(f"Error: Video file not found: {args.video_path}")
        return

    # Read first frame for point selection
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video: {args.video_path}")
        return

    ret, first_frame = cap.read()
    if not ret:
        print("Error: Cannot read first frame from video")
        cap.release()
        return

    height, width = first_frame.shape[:2]
    fps = int(cap.get(cv2.CAP_PROP_FPS)) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height} @ {fps}fps, {total_frames} frames")
    print()

    # Step 1: Let user select points on the first frame
    selector = PointSelector("Select Target", first_frame)
    points, labels = selector.wait_for_selection()

    if points is None:
        print("Tracking cancelled by user")
        cap.release()
        return

    # Step 2: Initialize predictor
    print("\nInitializing SAMURAI predictor...")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_cfg = determine_model_cfg(args.model_path)
    print(f"Device: {device}")
    print(f"Model config: {model_cfg}")
    print(f"Checkpoint: {args.model_path}")

    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device=str(device))

    # Step 3: Initialize video state
    print("\nInitializing video state...")
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        ),
    ):
        state = init_state(
            predictor, offload_video_to_cpu=True, offload_state_to_cpu=True, first_image=first_frame
        )
        # Add points to the first frame
        print("Adding initial points...")
        predictor.add_new_points_or_box(state, points=points, labels=labels, frame_idx=0, obj_id=0)

        # Step 4: Track through the video
        print("\n=== Starting Tracking ===")
        print("Press 'q' to quit, 'p' to pause, 'SPACE' to advance one frame when paused")
        print()

        if args.save_to_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(args.video_output_path, fourcc, 30, (width, height))

        paused = False
        frame_count = 0

        predictor.propagate_in_video_preflight(state)

        output_dict = state["output_dict"]
        obj_ids = state["obj_ids"]
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")

        meter = cv2.TickMeter()

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            meter.start()
            masks = process_video_frames(predictor, state, frame)

            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(obj_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            for obj_id, mask in mask_to_vis.items():
                mask_img = np.zeros((height, width, 3), np.uint8)
                mask_img[mask] = color[(obj_id + 1) % len(color)]
                frame = cv2.addWeighted(frame, 1, mask_img, 0.2, 0)

            for obj_id, bbox in bbox_to_vis.items():
                cv2.rectangle(
                    frame,
                    (bbox[0], bbox[1]),
                    (bbox[0] + bbox[2], bbox[1] + bbox[3]),
                    color[obj_id % len(color)],
                    2,
                )

            meter.stop()
            print(f"Inference time : {1000 * meter.getAvgTimeSec()} ms")
            cv2.putText(
                frame,
                f"{int(meter.getFPS())}",
                (width - 200, 80),
                cv2.FONT_HERSHEY_COMPLEX,
                3,
                (0, 255, 0),
                1,
            )
            # Display
            cv2.imshow("Tracking", frame)

            # Handle keyboard input
            if paused:
                key = cv2.waitKey(0) & 0xFF
            else:
                key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\nQuitting...")
                break
            elif key == ord("p"):
                paused = not paused
            elif key == 32 and paused:  # SPACE when paused
                paused = False

            # Optional: save frames
            if args.save_to_video:
                out.write(frame)

        print(f"\nTracking completed: {frame_count} frames processed")

    # Cleanup
    cv2.destroyAllWindows()
    cap.release()
    del predictor, state
    gc.collect()
    if torch.cuda.is_available():
        torch.clear_autocast_cache()
        torch.cuda.empty_cache()

    print("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive real-time tracking with SAMURAI")
    parser.add_argument(
        "--video_path", required=True, help="Path to input video file (.mp4, .avi, etc.)"
    )
    parser.add_argument(
        "--model_path",
        default="/home/wanghl/model_zoo/sam2/sam2.1_hiera_tiny.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--save_to_video", action="store_true", help="Save tracking result video to disk"
    )
    parser.add_argument(
        "--video_output_path", default="tracking_output.mp4", help="Output path for saved video"
    )
    args = parser.parse_args()

    main(args)
