#!/usr/bin/env python3
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def convert_record(record: dict) -> dict:
    messages = record.get("messages", [])
    if len(messages) < 2:
        raise ValueError("messages must contain user and assistant")
    prompt = messages[0]["content"]
    answer = messages[1]["content"]
    task_type = record.get("task_type", "")
    packed_answer = json.dumps({"task_type": task_type, "answer": answer}, ensure_ascii=False)
    return {
        "prompt": prompt,
        "answer": packed_answer,
        "images": record.get("images", []),
        "task_type": task_type,
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--train_output", required=True)
    parser.add_argument("--val_output", required=True)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    buckets: dict[str, list[dict]] = defaultdict(list)
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            item = convert_record(raw)
            buckets[item["task_type"]].append(item)

    rng = random.Random(args.seed)
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    for task_type, rows in sorted(buckets.items()):
        rng.shuffle(rows)
        val_n = max(1, int(len(rows) * args.val_ratio))
        val_rows.extend(rows[:val_n])
        train_rows.extend(rows[val_n:])
        print(f"{task_type}: total={len(rows)}, train={len(rows)-val_n}, val={val_n}")

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    write_jsonl(Path(args.train_output), train_rows)
    write_jsonl(Path(args.val_output), val_rows)
    print(f"train={len(train_rows)} -> {args.train_output}")
    print(f"val={len(val_rows)} -> {args.val_output}")


if __name__ == "__main__":
    main()
