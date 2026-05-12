from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "configs" / "luna16.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a trained YOLO checkpoint.")
    parser.add_argument("--weights", required=True, help="Path to the trained checkpoint, usually best.pt.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset yaml path.")
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.6, help="IoU threshold.")
    parser.add_argument("--device", default="cpu", help="Device string accepted by Ultralytics.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader worker count.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
