#!/usr/bin/env python3
"""Filter detector outputs and add normalized boxes for object-grounded prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes_root", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--min_area", type=float, default=0.01)
    parser.add_argument("--max_area", type=float, default=0.6)
    parser.add_argument("--edge_margin", type=float, default=0.01)
    return parser.parse_args()


def bbox_area_ratio(bbox: list[float], width: int, height: int) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1) / float(width * height)


def far_from_border(bbox: list[float], width: int, height: int, edge_margin: float) -> bool:
    x1, y1, x2, y2 = bbox
    return (
        x1 >= edge_margin * width
        and y1 >= edge_margin * height
        and x2 <= (1.0 - edge_margin) * width
        and y2 <= (1.0 - edge_margin) * height
    )


def normalize_bbox(bbox: list[float], width: int, height: int) -> list[int]:
    x1, y1, x2, y2 = bbox
    return [
        round(x1 / width * 1000),
        round(y1 / height * 1000),
        round(x2 / width * 1000),
        round(y2 / height * 1000),
    ]


def filter_file(path: Path, args: argparse.Namespace) -> None:
    output = path.with_name("detect_result.filtered.jsonl")
    with path.open(encoding="utf-8") as fin, output.open("w", encoding="utf-8") as fout:
        for line in fin:
            record = json.loads(line)
            width = record.get("width")
            height = record.get("height")
            if not width or not height:
                continue

            kept = []
            for det in record.get("detections", []):
                bbox = det["bbox"]
                if det.get("score", 0.0) < args.conf:
                    continue
                if not (args.min_area <= bbox_area_ratio(bbox, width, height) <= args.max_area):
                    continue
                if not far_from_border(bbox, width, height, args.edge_margin):
                    continue
                det = dict(det)
                det["bbox_norm"] = normalize_bbox(bbox, width, height)
                kept.append(det)

            record["detections"] = kept
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    for detect_file in args.scenes_root.rglob("detect_result.jsonl"):
        filter_file(detect_file, args)


if __name__ == "__main__":
    main()
