#!/usr/bin/env python3
"""Merge and lightly balance motion/object tasks for SFT and GRPO packaging."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion_tasks", required=True, type=Path)
    parser.add_argument("--object_tasks", required=True, type=Path)
    parser.add_argument("--balance_by", nargs="+", default=["task_type"])
    parser.add_argument("--sft_output", required=True, type=Path)
    parser.add_argument("--grpo_output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--grpo_per_task", type=int, default=125)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fin:
        return [json.loads(line) for line in fin]


def bucket_key(record: dict) -> tuple:
    return (record["task_type"],)


def stable_shuffle(records: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    copied = list(records)
    rng.shuffle(copied)
    return copied


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        for record in records:
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")


def build_grpo_subset(records: list[dict], per_task: int, seed: int) -> list[dict]:
    buckets = defaultdict(list)
    for record in stable_shuffle(records, seed):
        buckets[bucket_key(record)].append(record)

    subset = []
    for key in sorted(buckets):
        subset.extend(buckets[key][:per_task])
    return subset


def main() -> None:
    args = parse_args()
    motion = load_jsonl(args.motion_tasks)
    objects = load_jsonl(args.object_tasks)
    merged = stable_shuffle(motion + objects, args.seed)
    grpo_subset = build_grpo_subset(merged, args.grpo_per_task, args.seed)

    write_jsonl(args.sft_output, merged)
    write_jsonl(args.grpo_output, grpo_subset)


if __name__ == "__main__":
    main()
