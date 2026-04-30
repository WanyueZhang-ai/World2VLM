#!/usr/bin/env python3
"""Convert metric actions into the motion-plan records consumed by HY-WorldPlay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
TRANSLATION_STEP = 0.08
ROTATION_STEP = 3.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--action_plan", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    return parser.parse_args()


def quantize_action(name: str, value: float) -> list[float]:
    base = ROTATION_STEP if name.startswith("turn") else TRANSLATION_STEP
    sign = -1.0 if name in {"turn_left", "shift_left", "backward"} else 1.0
    remainder = abs(value)
    chunks = []
    while remainder > base + 1e-6:
        chunks.append(sign * base)
        remainder -= base
    if remainder > 1e-6:
        chunks.append(sign * round(remainder, 2))
    return chunks


def load_action_templates(path: Path) -> list[list[dict]]:
    if not path.exists():
        return [
            [{"name": "forward", "value": 0.4}],
            [{"name": "turn_right", "value": 30.0}],
            [{"name": "shift_right", "value": 1.2}, {"name": "forward", "value": 0.8}],
        ]

    plans = []
    with path.open(encoding="utf-8") as fin:
        for line in fin:
            record = json.loads(line)
            plans.append(record["actions"])
    return plans


def build_motion_record(anchor_path: Path, actions: list[dict]) -> dict:
    return {
        "scene_id": anchor_path.stem,
        "anchor_path": str(anchor_path),
        "actions": actions,
        "latent_chunks": [
            {"action": action, "latent_values": quantize_action(action["name"], float(action["value"]))}
            for action in actions
        ],
    }


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plans = load_action_templates(args.action_plan)
    output_path = args.output_dir / "motion_plan.jsonl"

    with output_path.open("w", encoding="utf-8") as fout:
        for image_path in iter_images(args.input_dir):
            for actions in plans:
                fout.write(json.dumps(build_motion_record(image_path, actions), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
