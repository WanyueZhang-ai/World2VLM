from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate undetect prompt data (A1-A4)")
    parser.add_argument("--data_root", type=str, default="svc_data_out")
    parser.add_argument("--templates", type=str, default="svc_dataset_gen/templates_v2.jsonl")
    parser.add_argument("--out_path", type=str, default="undetect_data.jsonl")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--k_per_traj", type=int, default=3)
    return parser


def _format_number(value: float) -> str:
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    text = f"{value:.3f}"
    return text.rstrip("0").rstrip(".")


def _parse_action(action: str) -> tuple[str, float]:
    try:
        name, raw_value = action.split("_", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid action string: {action}") from exc
    value = float(raw_value)
    if name == "TL":
        return ("yaw", value)
    if name == "TR":
        return ("yaw", -value)
    if name == "FWD":
        return ("z", value)
    if name == "BACK":
        return ("z", -value)
    if name == "LEFT":
        return ("x", value)
    if name == "RIGHT":
        return ("x", -value)
    raise ValueError(f"Unknown action type: {name}")


def _sum_actions(actions: list[str]) -> dict[str, float]:
    sums = {"yaw": 0.0, "x": 0.0, "z": 0.0}
    for action in actions:
        axis, value = _parse_action(action)
        sums[axis] += value
    return sums


def _move_action(axis: str, signed_value: float) -> dict[str, Any] | None:
    if abs(signed_value) < 1e-6:
        return None
    if axis == "z":
        direction = "forward" if signed_value > 0 else "backward"
    elif axis == "x":
        direction = "left" if signed_value > 0 else "right"
    else:
        return None
    dist = abs(signed_value)
    return {
        "kind": "move",
        "axis": axis,
        "signed": signed_value,
        "dir": direction,
        "value": dist,
        "text": f"move {direction} {_format_number(dist)} meters",
    }


def _turn_action(signed_value: float) -> dict[str, Any] | None:
    if abs(signed_value) < 1e-6:
        return None
    direction = "left" if signed_value > 0 else "right"
    angle = abs(signed_value)
    return {
        "kind": "turn",
        "axis": "yaw",
        "signed": signed_value,
        "dir": direction,
        "value": angle,
        "text": f"turn {direction} {_format_number(angle)} degrees",
    }


def _action_from_token(token: str) -> dict[str, Any] | None:
    axis, value = _parse_action(token)
    if axis == "yaw":
        return _turn_action(value)
    return _move_action(axis, value)


def _actions_from_prefix(prefix: list[str]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for token in prefix:
        action = _action_from_token(token)
        if action:
            actions.append(action)
    return actions


def _format_action_seq(actions: list[dict[str, Any]], sep: str = "; ") -> str:
    return sep.join(action["text"] for action in actions)


def _format_action_claim(actions: list[dict[str, Any]]) -> str:
    texts = [action["text"] for action in actions]
    if not texts:
        return ""
    if len(texts) == 1:
        return texts[0]
    if len(texts) == 2:
        return f"{texts[0]} and {texts[1]}"
    return ", ".join(texts[:-1]) + f" and {texts[-1]}"


def _action_seq_to_natural(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return ""
    prefixes = ["First", "Then", "Next", "Finally"]
    parts: list[str] = []
    for idx, action in enumerate(actions):
        prefix = prefixes[min(idx, len(prefixes) - 1)]
        parts.append(f"{prefix} {action['text']}")
    return ", ".join(parts) + "."


def _dir_choice(direction: str) -> str:
    mapping = {
        "forward": "A",
        "backward": "B",
        "left": "C",
        "right": "D",
    }
    return mapping[direction]


def _turn_choice(direction: str) -> str:
    return {"left": "A", "right": "B"}[direction]


def _tf_choice(tf: str) -> str:
    return {"true": "A", "false": "B"}[tf]


def _tf_natural(tf: str, claim_action: str) -> str:
    if tf == "true":
        return f'The claim "{claim_action}" matches the image pair.'
    return f'The claim "{claim_action}" does not match the image pair.'


def _parse_templates(path: Path) -> dict[str, list[dict[str, Any]]]:
    raw = path.read_text(encoding="utf-8")
    lines = [line for line in raw.splitlines() if not line.strip().startswith("//")]
    content = "\n".join(lines)

    objects: list[dict[str, Any]] = []
    start = None
    depth = 0
    in_string = False
    escape = False

    for idx, ch in enumerate(content):
        if in_string:
            if escape:
                escape = False
            else:
                if ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
            continue
        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            if depth == 0:
                start = idx
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunk = content[start : idx + 1].strip()
                if chunk:
                    objects.append(json.loads(chunk))
                start = None

    if depth != 0 or len(objects) < 4:
        raise ValueError(f"Failed to parse templates from {path}")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for obj in objects:
        task_type = obj.get("task_type")
        if not isinstance(task_type, str):
            continue
        grouped.setdefault(task_type, []).append(obj)
    missing = [task_type for task_type in ("A1", "A2", "A3", "A4") if not grouped.get(task_type)]
    if missing:
        raise ValueError(f"Missing template variants for: {', '.join(missing)}")
    return grouped


def _pick_template(
    templates: dict[str, list[dict[str, Any]]],
    task_type: str,
    rng: random.Random,
) -> dict[str, Any]:
    return rng.choice(templates[task_type])


def _fill_template(template: dict[str, Any], replacements: dict[str, str]) -> dict[str, Any]:
    data = json.loads(json.dumps(template))
    for msg in data.get("messages", []):
        content = msg.get("content")
        if not isinstance(content, str):
            continue
        for key, value in replacements.items():
            content = content.replace("{" + key + "}", value)
        msg["content"] = content
    images = []
    for img in data.get("images", []):
        for key, value in replacements.items():
            img = img.replace("{" + key + "}", value)
        images.append(img)
    return {"task_type": data.get("task_type", []), "messages": data.get("messages", []), "images": images}


def _even_indices(num_frames: int, k: int) -> list[int]:
    if num_frames <= 1 or k <= 0:
        return []
    max_idx = num_frames - 1
    target = min(k, max_idx)
    base = [
        int(round((i + 1) * max_idx / (target + 1))) for i in range(target)
    ]
    base = [min(max_idx, max(1, idx)) for idx in base]
    out: list[int] = []
    used: set[int] = set()
    for idx in base:
        if idx not in used:
            out.append(idx)
            used.add(idx)
            continue
        for delta in range(1, max_idx + 1):
            for cand in (idx + delta, idx - delta):
                if 1 <= cand <= max_idx and cand not in used:
                    out.append(cand)
                    used.add(cand)
                    break
            else:
                continue
            break
    for cand in range(1, max_idx + 1):
        if len(out) >= target:
            break
        if cand not in used:
            out.append(cand)
            used.add(cand)
    return out


def _choose_move(sums: dict[str, float]) -> dict[str, Any] | None:
    if abs(sums["z"]) >= abs(sums["x"]):
        return _move_action("z", sums["z"])
    return _move_action("x", sums["x"])


def _perturb_signed(signed: float, step: float, rng: random.Random) -> float:
    if abs(signed) < 1e-6:
        return step
    if rng.random() < 0.5:
        return -signed
    return signed + (step if signed > 0 else -step)


def _normalize_image_path(path: Path) -> str:
    return str(path.resolve())


def _traj_type_from_dir(traj_dir: Path, traj_params: dict[str, Any] | None = None) -> str | None:
    match = re.match(r"traj_(T[1-4])_", traj_dir.name)
    if match:
        return match.group(1)

    name = traj_dir.name
    params = traj_params or {}

    kind = params.get("kind")
    if kind == "multi_step" or name.startswith("traj_multi_step_"):
        return "T4"
    if kind == "single_step":
        action = str(params.get("action", ""))
        if action in {"turn_left", "turn_right"}:
            return "T1"
        if action in {"forward", "backward"}:
            return "T2"
        if action in {"shift_left", "shift_right", "left", "right"}:
            return "T3"

    if name.startswith("traj_turn_left_") or name.startswith("traj_turn_right_"):
        return "T1"
    if name.startswith("traj_forward_") or name.startswith("traj_backward_"):
        return "T2"
    if name.startswith("traj_shift_left_") or name.startswith("traj_shift_right_"):
        return "T3"
    return None


def _build_a1_a3_prompts(
    traj_type: str,
    frame_paths: list[str],
    actions_prefix: list[list[str]],
    templates: dict[str, list[dict[str, Any]]],
    rng: random.Random,
    k_per_traj: int,
) -> tuple[list[dict[str, Any]], set[int]]:
    prompts: list[dict[str, Any]] = []
    used: set[int] = set()
    max_idx = len(frame_paths) - 1
    if max_idx <= 0:
        return prompts, used

    candidates = _even_indices(len(frame_paths), k_per_traj)

    def add_prompt(idx: int) -> bool:
        if not frame_paths[0] or not frame_paths[idx]:
            return False
        sums = _sum_actions(actions_prefix[idx])
        if traj_type == "T1":
            turn = _turn_action(sums["yaw"])
            if not turn:
                return False
            prompts.append(
                _fill_template(
                    _pick_template(templates, "A2", rng),
                    {
                        "IMG_1": frame_paths[0],
                        "IMG_2": frame_paths[idx],
                        "TURN_DIR": turn["dir"],
                        "ANGLE": _format_number(turn["value"]),
                        "TURN_CHOICE": _turn_choice(turn["dir"]),
                    },
                )
            )
            used.add(idx)
            return True
        if traj_type in {"T2", "T3"}:
            move = _choose_move(sums)
            if not move:
                return False
            prompts.append(
                _fill_template(
                    _pick_template(templates, "A1", rng),
                    {
                        "IMG_1": frame_paths[0],
                        "IMG_2": frame_paths[idx],
                        "DIR": move["dir"],
                        "DIST": _format_number(move["value"]),
                        "DIR_CHOICE": _dir_choice(move["dir"]),
                    },
                )
            )
            used.add(idx)
            return True
        return False

    for idx in candidates:
        if idx <= max_idx and idx not in used:
            add_prompt(idx)

    target = min(k_per_traj, max_idx)
    if len(prompts) < target:
        for idx in range(1, max_idx + 1):
            if len(prompts) >= target:
                break
            if idx in used:
                continue
            if add_prompt(idx):
                continue

    return prompts, used


def _build_a4_candidates(
    traj_type: str,
    frame_paths: list[str],
    actions_prefix: list[list[str]],
    used_indices: set[int],
    traj_params: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for idx in range(1, len(frame_paths)):
        if idx in used_indices:
            continue
        if not frame_paths[0] or not frame_paths[idx]:
            continue
        sums = _sum_actions(actions_prefix[idx])
        if traj_type == "T1":
            action = _turn_action(sums["yaw"])
        elif traj_type in {"T2", "T3"}:
            action = _choose_move(sums)
        else:
            action = None
        if not action:
            continue
        candidates.append(
            {
                "img1": frame_paths[0],
                "img2": frame_paths[idx],
                "actions": [action],
                "traj_params": traj_params,
            }
        )
    return candidates


def _build_t4_prompts(
    frame_paths: list[str],
    actions_prefix: list[list[str]],
    templates: dict[str, list[dict[str, Any]]],
    traj_params: dict[str, Any],
    rng: random.Random,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    prompts: list[dict[str, Any]] = []
    a4_candidates: list[dict[str, Any]] = []
    if not frame_paths or not frame_paths[0]:
        return prompts, a4_candidates

    idx_two = 6
    idx_three = 9

    def _get_actions(idx: int, expected_len: int) -> list[dict[str, Any]] | None:
        if idx >= len(frame_paths):
            return None
        if not frame_paths[idx]:
            return None
        actions = _actions_from_prefix(actions_prefix[idx])
        if len(actions) != expected_len:
            return None
        return actions

    actions_two = _get_actions(idx_two, 2)
    actions_three = _get_actions(idx_three, 3)
    if not actions_two or not actions_three:
        return prompts, a4_candidates

    if rng.random() < 0.5:
        a3_idx, a3_actions = idx_two, actions_two
        a4_idx, a4_actions = idx_three, actions_three
    else:
        a3_idx, a3_actions = idx_three, actions_three
        a4_idx, a4_actions = idx_two, actions_two

    prompts.append(
        _fill_template(
            _pick_template(templates, "A3", rng),
            {
                "IMG_1": frame_paths[0],
                "IMG_2": frame_paths[a3_idx],
                "ACTION_SEQ": _format_action_seq(a3_actions),
                "ACTION_COUNT": str(len(a3_actions)),
                "ACTION_SEQ_NL": _action_seq_to_natural(a3_actions),
            },
        )
    )
    a4_candidates.append(
        {
            "img1": frame_paths[0],
            "img2": frame_paths[a4_idx],
            "actions": a4_actions,
            "traj_params": traj_params,
        }
    )
    return prompts, a4_candidates


def _make_false_action(
    action: dict[str, Any],
    traj_params: dict[str, Any],
    rng: random.Random,
) -> dict[str, Any] | None:
    signed = float(action["signed"])
    axis = action["axis"]
    if action["kind"] == "turn":
        step = float(traj_params.get("yaw_step_deg", 5.0))
        signed_false = _perturb_signed(signed, step, rng)
        return _turn_action(signed_false)
    step = float(traj_params.get("trans_step_m", 0.1))
    signed_false = _perturb_signed(signed, step, rng)
    return _move_action(axis, signed_false)


def _make_false_actions(
    actions: list[dict[str, Any]],
    traj_params: dict[str, Any],
    rng: random.Random,
) -> list[dict[str, Any]]:
    if not actions:
        return []
    pick = rng.randrange(len(actions))
    false_action = _make_false_action(actions[pick], traj_params, rng)
    if not false_action:
        return []
    out = list(actions)
    out[pick] = false_action
    return out


def main() -> None:
    args = build_arg_parser().parse_args()
    rng = random.Random(args.seed)

    data_root = Path(args.data_root).expanduser().resolve()
    templates = _parse_templates(Path(args.templates).expanduser().resolve())
    out_path = Path(args.out_path).expanduser().resolve()

    traj_paths = sorted(data_root.glob("*/traj_*/traj.json"))
    all_prompts: list[dict[str, Any]] = []
    a4_candidates: list[dict[str, Any]] = []

    for traj_json in traj_paths:
        payload = json.loads(traj_json.read_text(encoding="utf-8"))
        traj_dir = traj_json.parent
        traj_type = _traj_type_from_dir(traj_dir, payload.get("traj_params", {}))
        if not traj_type:
            continue
        actions_prefix = payload.get("actions_prefix", [])
        frames = payload.get("transforms", {}).get("frames", [])
        if not actions_prefix or not frames:
            continue

        count = min(len(actions_prefix), len(frames))
        if count < 2:
            continue
        actions_prefix = actions_prefix[:count]
        frames = frames[:count]

        frame_paths: list[str] = []
        for frame in frames:
            file_path = frame.get("file_path")
            if not file_path:
                frame_paths.append("")
                continue
            resolved = traj_dir / file_path.lstrip("./")
            frame_paths.append(
                _normalize_image_path(resolved)
            )

        if not frame_paths[0]:
            continue

        traj_params = payload.get("traj_params", {})
        if traj_type == "T4":
            prompts, t4_candidates = _build_t4_prompts(
                frame_paths, actions_prefix, templates, traj_params, rng
            )
            all_prompts.extend(prompts)
            a4_candidates.extend(t4_candidates)
            continue

        prompts, used = _build_a1_a3_prompts(
            traj_type,
            frame_paths,
            actions_prefix,
            templates,
            rng,
            args.k_per_traj,
        )
        all_prompts.extend(prompts)
        candidates = _build_a4_candidates(
            traj_type, frame_paths, actions_prefix, used, traj_params, rng
        )
        if candidates:
            a4_candidates.append(rng.choice(candidates))

    rng.shuffle(a4_candidates)
    for idx, candidate in enumerate(a4_candidates):
        is_true = idx % 2 == 0
        actions = candidate["actions"]
        if is_true:
            claim_action = _format_action_claim(actions)
            tf = "true"
        else:
            false_actions = _make_false_actions(actions, candidate["traj_params"], rng)
            if not false_actions:
                continue
            claim_action = _format_action_claim(false_actions)
            tf = "false"
        if not claim_action:
            continue
        all_prompts.append(
            _fill_template(
                _pick_template(templates, "A4", rng),
                {
                    "IMG_1": candidate["img1"],
                    "IMG_2": candidate["img2"],
                    "CLAIM_ACTION": claim_action,
                    "TF": tf,
                    "TF_CHOICE": _tf_choice(tf),
                    "TF_NL": _tf_natural(tf, claim_action),
                },
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for item in all_prompts:
            f.write(json.dumps(item, ensure_ascii=True) + "\n")
    print(f"Wrote {len(all_prompts)} records to {out_path}")


if __name__ == "__main__":
    main()
