#!/usr/bin/env python3
"""Build D1-D4 object-grounded WorldVLM tasks from filtered detector tracks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes_root", required=True, type=Path)
    parser.add_argument("--pairing", default="max_displacement")
    parser.add_argument("--task_types", nargs="+", default=["D1", "D2", "D3", "D4"])
    parser.add_argument("--templates", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_json(path: Path):
    with path.open(encoding="utf-8") as fin:
        return json.load(fin)


def load_detect_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fin:
        return [json.loads(line) for line in fin]


def action_text(actions: list[dict]) -> str:
    parts = []
    for action in actions:
        unit = "degrees" if action["name"].startswith("turn") else "meters"
        parts.append(f"{action['name'].replace('_', ' ')} {action['value']} {unit}")
    return "; ".join(parts)


def best_pair(rows: list[dict]) -> tuple[dict, dict] | None:
    if len(rows) < 2:
        return None

    start = rows[0]
    end = rows[-1]
    start_by_track = {det["track_id"]: det for det in start["detections"]}
    end_by_track = {det["track_id"]: det for det in end["detections"]}

    for track_id in sorted(set(start_by_track) & set(end_by_track)):
        return start_by_track[track_id], end_by_track[track_id]
    return None


def build_records(scene_dir: Path, traj_dir: Path, task_types: list[str]) -> list[dict]:
    traj = load_json(traj_dir / "traj.json")
    detect_file = traj_dir / "detect_result.filtered.jsonl"
    if not detect_file.exists():
        detect_file = traj_dir / "detect_result.jsonl"

    rows = load_detect_rows(detect_file)
    pair = best_pair(rows)
    if pair is None:
        return []

    start_det, end_det = pair
    motion = action_text(traj["actions"])
    label = start_det["label"]
    start_box = start_det.get("bbox_norm", start_det["bbox"])
    end_box = end_det.get("bbox_norm", end_det["bbox"])
    start_image = f"./{(traj_dir / traj['sample_paths'][0]).relative_to(scene_dir.parent)}"
    end_image = f"./{(traj_dir / traj['sample_paths'][-1]).relative_to(scene_dir.parent)}"

    records = {
        "D1": {
            "messages": [
                {
                    "role": "user",
                    "content": f"<image><image>In the first image, the {label} is at bbox {start_box}. "
                    "Bboxes use normalized integer coordinates in [0,1000]. "
                    f"After the camera does \"{motion}\", give the bbox of the same {label} in the second image. "
                    "Answer with bbox [x1, y1, x2, y2] only.",
                },
                {"role": "assistant", "content": str(end_box)},
            ],
            "images": [start_image, end_image],
        },
        "D2": {
            "messages": [
                {
                    "role": "user",
                    "content": f"<image>In the image, the {label} is at bbox {start_box}. "
                    f"After the camera does \"{motion}\", state the visibility result in one short sentence.",
                },
                {"role": "assistant", "content": f"The {label} remains visible."},
            ],
            "images": [start_image],
        },
        "D3": {
            "messages": [
                {
                    "role": "user",
                    "content": f"<image><image>The {label} in the first image (bbox {start_box}) and the {label} "
                    f"in the second image (bbox {end_box}) are the same physical object. "
                    "Explain the camera motion in one concise sentence.",
                },
                {"role": "assistant", "content": f"First {motion}."},
            ],
            "images": [start_image, end_image],
        },
        "D4": {
            "messages": [
                {
                    "role": "user",
                    "content": f"<image><image>In the first image, the {label} is at bbox {start_box}. "
                    f"After the camera does \"{motion}\", the second image shows a {label} at bbox {end_box}. "
                    "Answer in one short sentence whether they refer to the same instance.",
                },
                {"role": "assistant", "content": "They are the same physical object instance."},
            ],
            "images": [start_image, end_image],
        },
    }
    return [{"task_type": task, **records[task]} for task in task_types]


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fout:
        for scene_dir in sorted(path for path in args.scenes_root.iterdir() if path.is_dir()):
            for traj_dir in sorted(path for path in scene_dir.iterdir() if path.is_dir() and path.name.startswith("traj_")):
                for record in build_records(scene_dir, traj_dir, args.task_types):
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
