import argparse
import os
import time

import cv2
import numpy as np
from samurai_utils import SamuraiTracker
from tqdm import tqdm

colors = [
    (0, 0, 255),  # red 0
    (0, 255, 0),  # green 1
    (255, 0, 0),  # blue 2
    (255, 255, 0),  # cyan 3
    (255, 0, 255),  # magenta 4
    (0, 255, 255),  # yellow 5
    (255, 255, 255),  # white 6
    (128, 128, 128),  # gray 7
    (140, 140, 0),  # mars green 8
    (167, 47, 0),  # klein blue 9
    (39, 88, 232),  # hermes orange 10
    (32, 0, 128),  # burgundy 11
    (208, 216, 129),  # tiffany blue 12
    (9, 0, 76),  # bordeaux 13
    (36, 220, 249),  # sennelier yellow 14
]


class PointSelector:
    """Handle mouse clicks to select points on the first frame."""

    def __init__(self, window_name, image):
        self.window_name = window_name
        self.image = image.copy()
        self.points = []

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        # Display instructions
        display_img = self.image.copy()
        cv2.putText(
            display_img,
            "Click to select target left click",
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
            self._update_display()

    def _update_display(self):
        """Update the display with clicked points."""
        display_img = self.image.copy()

        # Draw clicked points
        for point in self.points:
            color = (0, 255, 0)
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
            "Left click to add point",
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

        return points


class BoxSelector:
    """Handle mouse drag to draw bounding boxes on the first frame."""

    def __init__(self, window_name: str, image: np.ndarray):
        self.window_name = window_name
        self.image = image.copy()
        self.boxes: list = []  # 存储 tlwh 格式的框: [left, top, width, height]

        # 画框时的临时状态
        self.is_drawing = False
        self.start_x = -1
        self.start_y = -1
        self.current_x = -1
        self.current_y = -1

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, self.mouse_callback)
        self._show_instructions()

    def _show_instructions(self):
        """Display initial instructions."""
        display_img = self.image.copy()
        cv2.putText(
            display_img,
            "Hold left mouse to draw box | Right click: undo last",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            display_img,
            "SPACE: start | ESC: exit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )
        cv2.imshow(self.window_name, display_img)

    def mouse_callback(self, event, x, y, _flags, _param):
        """Handle mouse events for box drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # 左键按下：开始画框
            self.is_drawing = True
            self.start_x = x
            self.start_y = y
            self.current_x = x
            self.current_y = y
            self._update_display()

        elif event == cv2.EVENT_MOUSEMOVE:
            # 鼠标移动：更新当前框的终点
            if self.is_drawing:
                self.current_x = x
                self.current_y = y
                self._update_display()

        elif event == cv2.EVENT_LBUTTONUP:
            # 左键松开：完成画框
            if self.is_drawing:
                self.is_drawing = False
                # 计算 tlwh 格式（确保宽高为正）
                x1 = min(self.start_x, self.current_x)
                y1 = min(self.start_y, self.current_y)
                x2 = max(self.start_x, self.current_x)
                y2 = max(self.start_y, self.current_y)
                w = x2 - x1
                h = y2 - y1
                # 只添加有效框（宽高大于0）
                if w > 0 and h > 0:
                    self.boxes.append([x1, y1, x2, y2])
                self._update_display()

        elif event == cv2.EVENT_RBUTTONDOWN:
            # 右键按下：撤销最后一个框
            if self.boxes:
                self.boxes.pop()
                self._update_display()

    def _update_display(self):
        """Update display with drawn boxes."""
        display_img = self.image.copy()

        # 绘制所有已确认的框（绿色）
        for box in self.boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # 绘制当前正在画的临时框（红色）
        if self.is_drawing:
            x1 = min(self.start_x, self.current_x)
            y1 = min(self.start_y, self.current_y)
            x2 = max(self.start_x, self.current_x)
            y2 = max(self.start_y, self.current_y)
            cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        # 显示提示信息
        info = f"Boxes: {len(self.boxes)} | Right click: undo | SPACE: start, ESC: exit"
        cv2.putText(display_img, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow(self.window_name, display_img)

    def wait_for_selection(self) -> np.ndarray | None:
        """Wait for user to complete box selection."""
        print("\n=== Box Selection ===")
        print("Hold left mouse to draw bounding box")
        print("Right click: undo last box")
        print("SPACE: start tracking | ESC: exit\n")

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyWindow(self.window_name)
                return None
            elif key == 32:  # SPACE
                if len(self.boxes) > 0:
                    break
                print("Please draw at least one box!")

        cv2.destroyWindow(self.window_name)

        # 转换为 numpy 数组，tlwh 格式
        boxes = np.array(self.boxes, dtype=np.float32)
        print(f"Selected {len(boxes)} box(es) (tlwh format)")
        for i, box in enumerate(boxes):
            print(
                f"  Box {i + 1}: [left={box[0]:.0f}, top={box[1]:.0f}, right={box[2]:.0f}, bottom={box[3]:.0f}]"
            )

        return boxes


def main(args):
    tracker = SamuraiTracker(args.model_path, use_fp16=args.use_fp16)

    cap = cv2.VideoCapture(args.video_path)
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tracker.init_video(frame_width, frame_height)

    # fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # out = cv2.VideoWriter('./trt_demo.mp4', fourcc, 30, (frame_width, frame_height))

    name_window = os.path.basename(args.video_path)
    cv2.namedWindow(name_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name_window, 960, 720)

    start = time.time()
    for frame_idx in tqdm(range(num_frames), desc="Processing video frames"):
        # print(f"\033[93mframe_idx: {frame_idx}\033[0m")
        # start = time.time()
        ret, frame = cap.read()
        if not ret:
            raise ValueError("Failed to read frame from video.")

        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if frame_idx == 0:
            # bbox = cv2.selectROI(name_window, frame) # (x, y, w, h)
            # x, y, w, h = bbox
            # first_frame_bbox = [x, y, x + w, y + h]

            # first_frame_bbox = load_txt(args.txt_path)[0][0]
            bbox_selector = (
                PointSelector(name_window, frame)
                if args.use_point
                else BoxSelector(name_window, frame)
            )
            first_frame_coords = bbox_selector.wait_for_selection()
            if first_frame_coords is None:
                break

            mask = (
                tracker.add_first_frame_points(input_image, first_frame_coords)
                if args.use_point
                else tracker.add_first_frame_bbox(input_image, first_frame_coords)
            )
        else:
            mask = tracker.track_step(input_image)

        mask = mask > 0.0
        non_zero_indices = np.argwhere(mask)
        if len(non_zero_indices) == 0:
            bbox = [0, 0, 0, 0]
        else:
            y_min, x_min = non_zero_indices.min(axis=0).tolist()
            y_max, x_max = non_zero_indices.max(axis=0).tolist()
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        mask_img = np.zeros((frame_height, frame_width, 3), np.uint8)
        mask_img[mask] = colors[1]
        frame = cv2.addWeighted(frame, 1, mask_img, 0.4, 0)

        cv2.rectangle(
            frame, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), colors[1], 2
        )

        cv2.imshow(name_window, frame)
        cv2.waitKey(1)

    elapsed = (time.time() - start) * 1000
    print(f"Elapsed time: {elapsed:.3f} ms")
    print(f"every frame spend time: {elapsed / (frame_idx + 1):.2f}ms")
    print(f"fps: {1000 / (elapsed / (frame_idx + 1)):.2f}")

    cap.release()
    # out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video_path",
        default="../assets/1917.mp4",
        help="Input video path or directory of frames.",
    )
    parser.add_argument(
        "--model_path",
        default="/home/wanghl/model_zoo/sam2/sam2.1_hiera_tiny.pt",
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--video_output_path", default="demo.mp4", help="Path to save the output video."
    )
    parser.add_argument("--save_to_video", default=False, help="Save results to a video.")
    parser.add_argument("--use_fp16", action="store_true", help="Use FP16 precision for inference.")
    parser.add_argument(
        "-p", "--use_point", action="store_true", help="Use FP16 precision for inference."
    )
    args = parser.parse_args()

    main(args)
