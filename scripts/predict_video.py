from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs" / "predict-video"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on a webcam or video file.")
    parser.add_argument("--weights", required=True, help="Checkpoint path.")
    parser.add_argument(
        "--source",
        default="0",
        help="Video path or camera id. Use 0 for the default webcam.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--device", default="cpu", help="Device string accepted by Ultralytics.")
    parser.add_argument("--project", default=str(DEFAULT_OUTPUT.parent), help="Prediction output root.")
    parser.add_argument("--name", default=DEFAULT_OUTPUT.name, help="Prediction run name.")
    parser.add_argument("--show", action="store_true", help="Display the rendered result.")
    return parser.parse_args()


def normalize_source(source: str) -> int | str:
    return int(source) if source.isdigit() else source


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=normalize_source(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        save=True,
        project=args.project,
        name=args.name,
        show=args.show,
    )


if __name__ == "__main__":
    main()
