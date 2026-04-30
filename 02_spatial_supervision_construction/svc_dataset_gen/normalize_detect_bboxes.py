from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from PIL import Image

BBOX_RE = re.compile(r"bbox\s*\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize bbox in detect jsonl prompts to integer [0,1000] without rerunning detection."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input detect jsonl file")
    parser.add_argument("--output", type=Path, required=True, help="Output jsonl file")
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input file in place (ignores --output, writes atomically).",
    )
    parser.add_argument(
        "--image-root",
        type=Path,
        default=None,
        help="Optional root to resolve relative image paths in records.",
    )
    parser.add_argument(
        "--skip-missing-images",
        action="store_true",
        help="Skip records with missing image files instead of failing.",
    )
    return parser.parse_args()


def _resolve_image_path(raw: str, image_root: Path | None) -> Path:
    p = Path(raw)
    if p.is_absolute():
        return p
    if image_root is not None:
        return (image_root / p).resolve()
    return p.resolve()


def _get_image_size(path: Path, cache: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    key = str(path)
    if key in cache:
        return cache[key]
    with Image.open(path) as im:
        w, h = im.size
    cache[key] = (w, h)
    return w, h


def _norm_1000(x: float, y: float, w: int, h: int) -> Tuple[int, int]:
    if w <= 0 or h <= 0:
        return 0, 0
    nx = max(0, min(1000, int(round(x / float(w) * 1000.0))))
    ny = max(0, min(1000, int(round(y / float(h) * 1000.0))))
    return nx, ny


def _replace_bboxes_in_text(text: str, dims_by_occurrence: Sequence[Tuple[int, int]]) -> Tuple[str, int]:
    matches = list(BBOX_RE.finditer(text))
    if not matches:
        return text, 0

    parts: List[str] = []
    cursor = 0
    changed = 0
    for i, m in enumerate(matches):
        parts.append(text[cursor : m.start()])
        x1 = float(m.group(1))
        y1 = float(m.group(2))
        x2 = float(m.group(3))
        y2 = float(m.group(4))

        w, h = dims_by_occurrence[min(i, len(dims_by_occurrence) - 1)]
        nx1, ny1 = _norm_1000(x1, y1, w, h)
        nx2, ny2 = _norm_1000(x2, y2, w, h)

        replaced = f"bbox [{nx1}, {ny1}, {nx2}, {ny2}]"
        parts.append(replaced)
        cursor = m.end()

        if replaced != m.group(0):
            changed += 1

    parts.append(text[cursor:])
    return "".join(parts), changed


def _collect_dims_for_record(record: Dict[str, Any], image_root: Path | None, size_cache: Dict[str, Tuple[int, int]]) -> List[Tuple[int, int]]:
    images = record.get("images", [])
    if not isinstance(images, list) or len(images) == 0:
        raise ValueError("Record has no images array, cannot normalize bbox.")

    dims: List[Tuple[int, int]] = []
    for img in images:
        if not isinstance(img, str):
            raise ValueError(f"Invalid image path: {img}")
        img_path = _resolve_image_path(img, image_root)
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        dims.append(_get_image_size(img_path, size_cache))
    return dims


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            s = line.strip()
            if not s:
                continue
            yield ln, json.loads(s)


def main() -> None:
    args = _parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input not found: {args.input}")

    out_path = args.input if args.in_place else args.output
    if args.in_place:
        tmp_path = args.input.with_suffix(args.input.suffix + ".tmp")
    else:
        tmp_path = out_path

    size_cache: Dict[str, Tuple[int, int]] = {}
    total = 0
    changed_records = 0
    changed_boxes = 0
    skipped = 0

    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    with tmp_path.open("w", encoding="utf-8") as out_f:
        for ln, record in _iter_jsonl(args.input):
            total += 1
            try:
                dims = _collect_dims_for_record(record, args.image_root, size_cache)
            except FileNotFoundError:
                if args.skip_missing_images:
                    skipped += 1
                    out_f.write(json.dumps(record, ensure_ascii=True) + "\n")
                    continue
                raise

            msgs = record.get("messages", [])
            record_changed_boxes = 0
            if isinstance(msgs, list):
                for msg in msgs:
                    if not isinstance(msg, dict):
                        continue
                    content = msg.get("content")
                    if not isinstance(content, str):
                        continue
                    new_content, c = _replace_bboxes_in_text(content, dims)
                    if c > 0:
                        msg["content"] = new_content
                        record_changed_boxes += c

            if record_changed_boxes > 0:
                changed_records += 1
                changed_boxes += record_changed_boxes
            out_f.write(json.dumps(record, ensure_ascii=True) + "\n")

    if args.in_place:
        tmp_path.replace(args.input)

    print(
        f"Done. total={total}, changed_records={changed_records}, changed_boxes={changed_boxes}, skipped={skipped}, cache_images={len(size_cache)}"
    )
    print(f"Output: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
