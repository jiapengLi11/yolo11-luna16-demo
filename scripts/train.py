from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "configs" / "luna16.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a YOLO model on the LUNA16-style dataset.")
    parser.add_argument("--model", default="yolo11n.pt", help="Path or model name used for initialization.")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Dataset yaml path.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size.")
    parser.add_argument("--batch", type=int, default=16, help="Batch size.")
    parser.add_argument("--device", default="cpu", help="Device string accepted by Ultralytics.")
    parser.add_argument("--workers", type=int, default=0, help="Dataloader worker count.")
    parser.add_argument("--project", default=str(ROOT / "runs"), help="Output directory for runs.")
    parser.add_argument("--name", default="luna16-yolo11n", help="Run name.")
    parser.add_argument("--resume", action="store_true", help="Resume from the last checkpoint if available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        resume=args.resume,
        cache=True,
        amp=False,
    )


if __name__ == "__main__":
    main()
