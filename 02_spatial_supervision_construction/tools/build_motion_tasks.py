#!/usr/bin/env python3
"""Build the four motion-centric WorldVLM tasks from trajectory bundles."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenes_root", required=True, type=Path)
    parser.add_argument("--pairing", default="max_displacement")
    parser.add_argument("--task_types", nargs="+", default=["A1", "A2", "A3", "A4"])
    parser.add_argument("--templates", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_traj(path: Path) -> dict:
    with path.open(encoding="utf-8") as fin:
        return json.load(fin)


def image_pair_for_max_displacement(traj: dict) -> list[str]:
    sample_paths = traj["sample_paths"]
    return [sample_paths[0], sample_paths[-1]]


def action_text(traj: dict) -> str:
    parts = []
    for action in traj["actions"]:
        unit = "degrees" if action["name"].startswith("turn") else "meters"
        parts.append(f"{action['name'].replace('_', ' ')} {action['value']} {unit}")
    return "; ".join(parts)


def build_records(scene_dir: Path, traj_dir: Path, task_types: list[str]) -> list[dict]:
    traj = load_traj(traj_dir / "traj.json")
    image_a, image_b = image_pair_for_max_displacement(traj)
    rel_a = str((traj_dir / image_a).relative_to(scene_dir.parent))
    rel_b = str((traj_dir / image_b).relative_to(scene_dir.parent))
    motion_text = action_text(traj)

    task_map = {
        "A1": {
            "prompt": "<image><image>Estimate the translation distance between the two views. "
            "Answer using the format: move left/right/forward/backward X meters.",
            "answer": motion_text if len(traj["actions"]) == 1 and not traj["actions"][0]["name"].startswith("turn") else None,
        },
        "A2": {
            "prompt": "<image><image>Estimate the rotation angle between the two views. "
            "Answer using the format: turn left/right X degrees.",
            "answer": motion_text if len(traj["actions"]) == 1 and traj["actions"][0]["name"].startswith("turn") else None,
        },
        "A3": {
            "prompt": "<image><image>Infer the ordered action sequence connecting the two views. "
            "Use ';' as the separator.",
            "answer": motion_text,
        },
        "A4": {
            "prompt": f"<image><image>True or false: the camera did \"{motion_text}\" to get the second image.",
            "answer": "true",
        },
    }

    records = []
    for task_type in task_types:
        spec = task_map[task_type]
        if spec["answer"] is None:
            continue
        records.append(
            {
                "task_type": task_type,
                "messages": [
                    {"role": "user", "content": spec["prompt"]},
                    {"role": "assistant", "content": spec["answer"]},
                ],
                "images": [f"./{rel_a}", f"./{rel_b}"],
            }
        )
    return records


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
