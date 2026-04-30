#!/usr/bin/env python3
"""Build an anonymized anchor-image manifest for world-model generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--source_domain",
        default="mixed",
        choices=["mixed", "real_scene", "simulated_scene"],
        help="Domain tag used later for balanced packaging.",
    )
    return parser.parse_args()


def iter_images(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS
    )


def build_entry(path: Path, source_domain: str) -> dict:
    return {
        "scene_id": path.stem,
        "anchor_path": str(path),
        "source_domain": source_domain,
    }


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fout:
        for image_path in iter_images(args.input_dir):
            fout.write(json.dumps(build_entry(image_path, args.source_domain), ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
