"""
01_people_detection_camera.py

Realtime people detection from camera/video stream using Ultralytics YOLO.
- Detects class 0 (person) only
- Draws bounding boxes and confidence
- Optional recording of annotated output
"""

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def parse_source(source: str):
    """Interpret numeric values as camera index, otherwise use as path/URL."""
    return int(source) if source.isdigit() else source


def parse_args():
    parser = argparse.ArgumentParser(description="Ultralytics people detection from camera stream")
    parser.add_argument("--source", default="0", help="Camera index (e.g. 0) or video path/URL")
    parser.add_argument("--model", default="yolov8n.pt", help="Ultralytics model path or name")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--device", default=None, help="Inference device (e.g. cpu, 0, cuda:0)")
    parser.add_argument("--save", default=None, help="Optional output video path (e.g. out.mp4)")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames (0 = no limit)")
    parser.add_argument("--hide-window", action="store_true", help="Disable OpenCV display window")
    return parser.parse_args()


def create_video_writer(save_path: str, width: int, height: int, fps: float):
    if not save_path:
        return None
    output_path = Path(save_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    use_fps = fps if fps and fps > 0 else 30.0
    return cv2.VideoWriter(str(output_path), fourcc, use_fps, (width, height))


def main():
    args = parse_args()

    source = parse_source(args.source)
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {args.source}")

    model = YOLO(args.model)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    writer = create_video_writer(args.save, width, height, fps)

    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = model.predict(
                frame,
                conf=args.conf,
                classes=[0],
                device=args.device,
                verbose=False,
            )

            result = results[0]
            person_count = len(result.boxes) if result.boxes is not None else 0
            annotated = result.plot()

            cv2.putText(
                annotated,
                f"People: {person_count}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            if writer is not None:
                writer.write(annotated)

            if not args.hide_window:
                cv2.imshow("Ultralytics People Detector", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

            frame_count += 1
            if args.max_frames > 0 and frame_count >= args.max_frames:
                break

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if not args.hide_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

