from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from ultralytics.utils.plotting import Annotator, colors


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "runs" / "sahi"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sliced inference with SAHI on a chest CT image.")
    parser.add_argument("--weights", required=True, help="Checkpoint path.")
    parser.add_argument("--source", required=True, help="Input image path.")
    parser.add_argument("--device", default="cpu", help="Torch device, for example cpu or cuda:0.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--slice-height", type=int, default=512, help="Slice height.")
    parser.add_argument("--slice-width", type=int, default=512, help="Slice width.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT / "prediction.jpg"), help="Rendered output path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(args.source)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {args.source}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path=args.weights,
        confidence_threshold=args.conf,
        device=args.device,
    )
    result = get_sliced_prediction(
        image,
        model,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=0.2,
        overlap_width_ratio=0.2,
    )

    annotator = Annotator(image.copy())
    for prediction in result.object_prediction_list:
        bbox = prediction.bbox
        annotator.box_label(
            [bbox.minx, bbox.miny, bbox.maxx, bbox.maxy],
            str(prediction.category.name),
            color=colors(int(prediction.category.id), True),
        )
    cv2.imwrite(str(output_path), annotator.result())


if __name__ == "__main__":
    main()
