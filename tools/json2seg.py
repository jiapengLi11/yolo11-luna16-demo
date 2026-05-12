from __future__ import annotations

import argparse
import json
from pathlib import Path


def convert_labelme_to_yolo_seg(input_dir: Path, output_dir: Path, class_names: list[str]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for json_path in sorted(input_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

        image_width = payload["imageWidth"]
        image_height = payload["imageHeight"]
        lines: list[str] = []

        for shape in payload.get("shapes", []):
            label = shape["label"]
            if label not in class_names:
                raise ValueError(f"Unknown label {label!r} in {json_path.name}")
            class_id = class_names.index(label)

            coords: list[str] = [str(class_id)]
            for x, y in shape["points"]:
                coords.append(f"{x / image_width:.6f}")
                coords.append(f"{y / image_height:.6f}")
            lines.append(" ".join(coords))

        output_path = output_dir / f"{json_path.stem}.txt"
        output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert LabelMe polygons to YOLO segmentation labels.")
    parser.add_argument("--input-dir", required=True, help="Directory containing LabelMe json files.")
    parser.add_argument("--output-dir", required=True, help="Directory where YOLO txt files will be written.")
    parser.add_argument("--classes", nargs="+", required=True, help="Ordered class names.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_labelme_to_yolo_seg(Path(args.input_dir), Path(args.output_dir), args.classes)


if __name__ == "__main__":
    main()
