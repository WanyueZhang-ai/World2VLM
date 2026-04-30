import json
import math
import re
from collections import Counter
from typing import Any, Optional

REWARD_NAME = "worldvlm_multitype"
REWARD_TYPE = "batch"

_SPACE_RE = re.compile(r"\s+")
_TRAIL_PUNC_RE = re.compile(r"[ \t\r\n\.,;:!?\"]+$")
_NUMBER_RE = r"(-?\d+(?:\.\d+)?)"
_MOVE_RE = re.compile(rf"move\s+(forward|backward|left|right)\s+{_NUMBER_RE}\s+meters?", re.IGNORECASE)
_TURN_RE = re.compile(rf"turn\s+(left|right)\s+{_NUMBER_RE}\s+degrees?", re.IGNORECASE)
_BBOX_ANY_RE = re.compile(
    rf"(?:bbox\s*)?\[\s*{_NUMBER_RE}\s*,\s*{_NUMBER_RE}\s*,\s*{_NUMBER_RE}\s*,\s*{_NUMBER_RE}\s*\]",
    re.IGNORECASE,
)
_BBOX_ONLY_RE = re.compile(
    rf"^\s*(?:bbox\s*)?\[\s*{_NUMBER_RE}\s*,\s*{_NUMBER_RE}\s*,\s*{_NUMBER_RE}\s*,\s*{_NUMBER_RE}\s*\]\s*$",
    re.IGNORECASE,
)


def _norm_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = _TRAIL_PUNC_RE.sub("", text)
    text = _SPACE_RE.sub(" ", text)
    return text


def _parse_bool(text: str) -> Optional[bool]:
    t = _norm_text(text)
    if t in {"yes", "true"}:
        return True
    if t in {"no", "false"}:
        return False
    return None


def _parse_actions(text: str) -> list[tuple[str, float, str]]:
    text = _norm_text(text)
    actions: list[tuple[int, str, float, str]] = []

    for m in _MOVE_RE.finditer(text):
        direction = m.group(1).lower()
        value = float(m.group(2))
        actions.append((m.start(), f"move_{direction}", value, "move"))

    for m in _TURN_RE.finditer(text):
        direction = m.group(1).lower()
        value = float(m.group(2))
        actions.append((m.start(), f"turn_{direction}", value, "turn"))

    actions.sort(key=lambda x: x[0])
    return [(a, v, t) for _, a, v, t in actions]


def _numeric_score(action_type: str, pred: float, gt: float) -> float:
    err = abs(pred - gt)
    if action_type == "turn":
        low, high = 5.0, 90.0
    else:
        low, high = 0.5, 5.0
    if err <= low:
        return 1.0
    if err >= high:
        return 0.0
    return 1.0 - (err - low) / (high - low)


def _load_gt(ground_truth: str) -> tuple[str, str]:
    gt = (ground_truth or "").strip()
    try:
        obj = json.loads(gt)
        if isinstance(obj, dict):
            return str(obj.get("task_type", "")).strip(), str(obj.get("answer", "")).strip()
    except Exception:
        pass
    return "", gt


def _parse_bbox(text: str) -> Optional[list[float]]:
    match = _BBOX_ANY_RE.search(text or "")
    if not match:
        return None
    try:
        return [float(match.group(i)) for i in range(1, 5)]
    except (TypeError, ValueError):
        return None


def _bbox_format_score(text: str) -> float:
    text = text or ""
    if _BBOX_ONLY_RE.fullmatch(text):
        return 1.0
    if _parse_bbox(text) is not None:
        return 0.4
    return 0.0


def _clip(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _canonicalize_bbox(box: list[float]) -> tuple[list[float], float]:
    x1, y1, x2, y2 = box
    geom_ok = 1.0 if (x1 < x2 and y1 < y2) else 0.0
    box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    overflow = sum(max(0.0, -v) + max(0.0, v - 1000.0) for v in box) / 4.0
    range_score = max(0.0, 1.0 - overflow / 200.0)
    clipped = [_clip(v, 0.0, 1000.0) for v in box]
    validity = 0.7 * geom_ok + 0.3 * range_score
    return clipped, validity


def _bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _bbox_iou(box1: list[float], box2: list[float]) -> float:
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = _bbox_area(box1) + _bbox_area(box2) - inter
    if union <= 0:
        return 0.0
    return inter / union


def _bbox_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def _safe_log_ratio(a: float, b: float) -> float:
    return abs(math.log(max(a, 1e-6) / max(b, 1e-6)))


def _score_bbox(pred_text: str, gt_answer: str) -> dict[str, float]:
    gt_bbox = _parse_bbox(gt_answer)
    if gt_bbox is None:
        return {
            "overall": 0.0,
            "format": 0.0,
            "validity": 0.0,
            "geometry": 0.0,
            "iou": 0.0,
            "center": 0.0,
            "size": 0.0,
            "l1": 0.0,
        }

    format_score = _bbox_format_score(pred_text)
    pred_bbox = _parse_bbox(pred_text)
    if pred_bbox is None:
        return {
            "overall": 0.20 * format_score,
            "format": format_score,
            "validity": 0.0,
            "geometry": 0.0,
            "iou": 0.0,
            "center": 0.0,
            "size": 0.0,
            "l1": 0.0,
        }

    gt_box, _ = _canonicalize_bbox(gt_bbox)
    pred_box, validity_score = _canonicalize_bbox(pred_bbox)

    iou_score = _bbox_iou(pred_box, gt_box)

    pred_cx, pred_cy = _bbox_center(pred_box)
    gt_cx, gt_cy = _bbox_center(gt_box)
    center_dist = math.hypot(pred_cx - gt_cx, pred_cy - gt_cy)
    gt_w = max(1.0, gt_box[2] - gt_box[0])
    gt_h = max(1.0, gt_box[3] - gt_box[1])
    gt_diag = math.hypot(gt_w, gt_h)
    center_tol = max(80.0, 0.6 * gt_diag)
    center_score = max(0.0, 1.0 - center_dist / center_tol)

    pred_w = max(1.0, pred_box[2] - pred_box[0])
    pred_h = max(1.0, pred_box[3] - pred_box[1])
    size_err = _safe_log_ratio(pred_w, gt_w) + _safe_log_ratio(pred_h, gt_h)
    size_score = max(0.0, 1.0 - size_err / 1.6)

    mean_abs_err = sum(abs(pred_box[i] - gt_box[i]) for i in range(4)) / 4.0
    l1_score = max(0.0, 1.0 - mean_abs_err / 180.0)

    geometry_score = 0.45 * iou_score + 0.20 * center_score + 0.20 * l1_score + 0.15 * size_score
    geometry_score *= 0.3 + 0.7 * validity_score
    overall = 0.20 * format_score + 0.15 * validity_score + 0.65 * geometry_score
    return {
        "overall": overall,
        "format": format_score,
        "validity": validity_score,
        "geometry": geometry_score,
        "iou": iou_score,
        "center": center_score,
        "size": size_score,
        "l1": l1_score,
    }


def _score_single_step(pred: str, gt_answer: str) -> dict[str, float]:
    gt_actions = _parse_actions(gt_answer)
    pred_actions = _parse_actions(pred)
    if not gt_actions:
        return {"overall": 0.0, "format": 0.0, "action": 0.0, "numeric": 0.0}

    gt_name, gt_val, gt_type = gt_actions[0]
    format_score = 1.0 if pred_actions else 0.0
    best_action, best_numeric = 0.0, 0.0

    for p_name, p_val, p_type in pred_actions:
        a = 1.0 if p_name == gt_name else 0.0
        n = _numeric_score(gt_type, p_val, gt_val) if a > 0 else 0.0
        if 0.35 * a + 0.55 * n > 0.35 * best_action + 0.55 * best_numeric:
            best_action, best_numeric = a, n

    overall = 0.10 * format_score + 0.35 * best_action + 0.55 * best_numeric
    return {"overall": overall, "format": format_score, "action": best_action, "numeric": best_numeric}


def _score_sequence(pred: str, gt_answer: str) -> dict[str, float]:
    gt_actions = _parse_actions(gt_answer)
    pred_actions = _parse_actions(pred)
    if not gt_actions:
        return {"overall": 0.0, "format": 0.0, "action": 0.0, "order": 0.0, "numeric": 0.0}

    format_score = 1.0 if pred_actions else 0.0
    gt_names = [x[0] for x in gt_actions]
    pred_names = [x[0] for x in pred_actions]

    gt_counter = Counter(gt_names)
    pred_counter = Counter(pred_names)
    inter = sum((gt_counter & pred_counter).values())
    action_score = inter / max(len(gt_names), len(pred_names), 1)

    pos_len = min(len(gt_actions), len(pred_actions))
    order_hits = sum(1 for i in range(pos_len) if gt_actions[i][0] == pred_actions[i][0])
    order_score = order_hits / len(gt_actions)

    numeric_scores = []
    for i in range(pos_len):
        g_name, g_val, g_type = gt_actions[i]
        p_name, p_val, _ = pred_actions[i]
        if g_name == p_name:
            numeric_scores.append(_numeric_score(g_type, p_val, g_val))
    numeric_score = sum(numeric_scores) / len(gt_actions) if numeric_scores else 0.0

    overall = 0.10 * format_score + 0.25 * action_score + 0.35 * order_score + 0.30 * numeric_score
    if len(pred_actions) > len(gt_actions):
        overall = max(0.0, overall - 0.03 * (len(pred_actions) - len(gt_actions)))
    return {
        "overall": overall,
        "format": format_score,
        "action": action_score,
        "order": order_score,
        "numeric": numeric_score,
    }


def _score_binary(pred: str, gt_answer: str) -> dict[str, float]:
    p = _parse_bool(pred)
    g = _parse_bool(gt_answer)
    if g is None:
        return {"overall": 0.0, "format": 0.0, "accuracy": 0.0}
    format_score = 1.0 if p is not None else 0.0
    acc = 1.0 if (p is not None and p == g) else 0.0
    overall = 0.20 * format_score + 0.80 * acc
    return {"overall": overall, "format": format_score, "accuracy": acc}


def compute_score(reward_inputs: list[dict[str, Any]]) -> list[dict[str, float]]:
    scores: list[dict[str, float]] = []
    for reward_input in reward_inputs:
        raw_response = (reward_input.get("response", "") or "").strip()
        response = _norm_text(raw_response)
        if len(raw_response) > 200:
            scores.append({"overall": -1.0, "format": 0.0})
            continue

        task_type, gt_answer = _load_gt(reward_input.get("ground_truth", ""))

        if task_type in {"A1", "A2"}:
            score = _score_single_step(response, gt_answer)
        elif task_type in {"A3", "D3"}:
            score = _score_sequence(response, gt_answer)
        elif task_type == "D1":
            score = _score_bbox(raw_response, gt_answer)
        elif task_type in {"A4", "D2", "D4"}:
            score = _score_binary(response, gt_answer)
        else:
            # Fallback: strict text match
            eq = 1.0 if response == _norm_text(gt_answer) and gt_answer else 0.0
            score = {"overall": eq, "format": eq, "accuracy": eq}

        scores.append(score)
    return scores
