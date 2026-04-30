from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from PIL import Image

from traj.action_space import Act, Action, canonicalize


# --------- Templates (D1/D2/D3/D4) ---------
D1_TEMPLATES = [
    {
        "task_type": "D1",
        "messages": [
            {
                "role": "user",
                "content": "<image><image>In the first image, the {OBJECT} is at bbox {BBOX_1}. Bboxes use normalized integer coordinates in [0,1000]. After the camera does \"{ACTION_SEQ}\", give the bbox of the same {OBJECT} in the second image. Answer with bbox [x1, y1, x2, y2] only.",
            },
            {"role": "assistant", "content": "{BBOX_2}"},
        ],
        "images": ["{IMG_1}", "{IMG_2}"],
    },
]

D2_TEMPLATES = [
    {
        "task_type": "D2",
        "messages": [
            {
                "role": "user",
                "content": "<image>In the image, the {OBJECT} is at bbox {BBOX_1}. After the camera does \"{ACTION_SEQ}\", does this {OBJECT} disappear from view? Answer with yes or no.",
            },
            {"role": "assistant", "content": "{YN}"},
        ],
        "images": ["{IMG_1}"],
    },
    {
        "task_type": "D2",
        "messages": [
            {
                "role": "user",
                "content": "<image>In the image, the {OBJECT} is at bbox {BBOX_1}. After the camera does \"{ACTION_SEQ}\", which visibility result is correct?\nA. the object disappears from view\nB. the object remains visible\nAnswer with A or B only.",
            },
            {"role": "assistant", "content": "{VIS_CHOICE}"},
        ],
        "images": ["{IMG_1}"],
    },
    {
        "task_type": "D2",
        "messages": [
            {
                "role": "user",
                "content": "<image>In the image, the {OBJECT} is at bbox {BBOX_1}. After the camera does \"{ACTION_SEQ}\", state the visibility result in one short sentence.",
            },
            {"role": "assistant", "content": "{VIS_NL}"},
        ],
        "images": ["{IMG_1}"],
    },
]

D3_TEMPLATES = [
    {
        "task_type": "D3",
        "messages": [
            {
                "role": "user",
                "content": "<image><image>The {OBJECT} in the first image (bbox {BBOX_1}) and the {OBJECT} in the second image (bbox {BBOX_2}) are the same physical object. The camera used 1 or 2 actions in order. Write the full action sequence using ';' as a separator. Allowed actions:\n- move forward/backward/left/right X meters\n- turn left/right X degrees",
            },
            {"role": "assistant", "content": "{ACTION_SEQ}"},
        ],
        "images": ["{IMG_1}", "{IMG_2}"],
    },
    {
        "task_type": "D3",
        "messages": [
            {
                "role": "user",
                "content": "<image><image>The {OBJECT} in the first image (bbox {BBOX_1}) and the {OBJECT} in the second image (bbox {BBOX_2}) are the same physical object. Infer the ordered camera actions and report the step count.",
            },
            {"role": "assistant", "content": "{ACTION_COUNT} steps | {ACTION_SEQ}"},
        ],
        "images": ["{IMG_1}", "{IMG_2}"],
    },
    {
        "task_type": "D3",
        "messages": [
            {
                "role": "user",
                "content": "<image><image>The {OBJECT} in the first image (bbox {BBOX_1}) and the {OBJECT} in the second image (bbox {BBOX_2}) are the same physical object. Explain the camera motion in one concise sentence.",
            },
            {"role": "assistant", "content": "{ACTION_SEQ_NL}"},
        ],
        "images": ["{IMG_1}", "{IMG_2}"],
    },
]

D4_TEMPLATES = [
    {
        "task_type": "D4",
        "messages": [
            {
                "role": "user",
                "content": "<image><image>In the first image, the {OBJECT} is at bbox {BBOX_1}. After the camera does \"{ACTION_SEQ}\", the second image shows a {OBJECT} at bbox {BBOX_2}. Are these the same physical object instance? Answer with yes or no.",
            },
            {"role": "assistant", "content": "{YN}"},
        ],
        "images": ["{IMG_1}", "{IMG_2}"],
    },
    {
        "task_type": "D4",
        "messages": [
            {
                "role": "user",
                "content": "<image><image>In the first image, the {OBJECT} is at bbox {BBOX_1}. After the camera does \"{ACTION_SEQ}\", the second image shows a {OBJECT} at bbox {BBOX_2}. Which identity judgment is correct?\nA. same physical object instance\nB. different object instance\nAnswer with A or B only.",
            },
            {"role": "assistant", "content": "{IDENTITY_CHOICE}"},
        ],
        "images": ["{IMG_1}", "{IMG_2}"],
    },
    {
        "task_type": "D4",
        "messages": [
            {
                "role": "user",
                "content": "<image><image>In the first image, the {OBJECT} is at bbox {BBOX_1}. After the camera does \"{ACTION_SEQ}\", the second image shows a {OBJECT} at bbox {BBOX_2}. Answer in one short sentence whether they refer to the same instance.",
            },
            {"role": "assistant", "content": "{IDENTITY_NL}"},
        ],
        "images": ["{IMG_1}", "{IMG_2}"],
    },
]


# --------- Data classes ---------
@dataclass
class Detection:
    frame_idx: int
    track_id: Optional[int]
    label_id: int
    label: str
    bbox: Tuple[float, float, float, float]
    score: float
    width: int
    height: int

    def bbox_norm_1000(self) -> Tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.bbox
        if self.width <= 0 or self.height <= 0:
            return (0, 0, 0, 0)
        nx1 = max(0, min(1000, int(round(x1 / self.width * 1000.0))))
        ny1 = max(0, min(1000, int(round(y1 / self.height * 1000.0))))
        nx2 = max(0, min(1000, int(round(x2 / self.width * 1000.0))))
        ny2 = max(0, min(1000, int(round(y2 / self.height * 1000.0))))
        return (nx1, ny1, nx2, ny2)


@dataclass
class TrajMeta:
    traj_id: str
    actions_step: List[List[str]]
    action_substeps: int
    frame_paths: List[Path]


# --------- Action helpers ---------
_ACT_SHORT_TO_ENUM = {
    "TL": Act.TL,
    "TR": Act.TR,
    "FWD": Act.FWD,
    "BACK": Act.BACK,
    "LEFT": Act.LEFT,
    "RIGHT": Act.RIGHT,
}

_ACT_TO_TEXT = {
    Act.TL: "turn left",
    Act.TR: "turn right",
    Act.FWD: "move forward",
    Act.BACK: "move backward",
    Act.LEFT: "move left",
    Act.RIGHT: "move right",
}


def _parse_action(text: str) -> Action:
    if "_" not in text:
        raise ValueError(f"Invalid action string: {text}")
    short, raw_val = text.split("_", 1)
    act = _ACT_SHORT_TO_ENUM[short]
    return Action(act, float(raw_val))


def _format_value(val: float) -> str:
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    return f"{val:.3f}".rstrip("0").rstrip(".")


def actions_between(actions_step: List[List[str]], start_idx: int, end_idx: int) -> List[Action]:
    merged: List[Action] = []
    for i in range(start_idx + 1, end_idx + 1):
        merged.extend(_parse_action(a) for a in actions_step[i])
    return canonicalize(merged)


def actions_to_text(actions: Sequence[Action]) -> str:
    parts = []
    for act in actions:
        unit = "degrees" if act.act in {Act.TL, Act.TR} else "meters"
        parts.append(f"{_ACT_TO_TEXT[act.act]} {_format_value(act.value)} {unit}")
    return "; ".join(parts)


def actions_to_natural_text(actions: Sequence[Action]) -> str:
    if not actions:
        return ""
    prefixes = ["First", "Then", "Next", "Finally"]
    parts = []
    for idx, act in enumerate(actions):
        prefix = prefixes[min(idx, len(prefixes) - 1)]
        unit = "degrees" if act.act in {Act.TL, Act.TR} else "meters"
        parts.append(f"{prefix} {_ACT_TO_TEXT[act.act]} {_format_value(act.value)} {unit}")
    return ", ".join(parts) + "."


def _pick_template(templates: Sequence[Dict[str, Any]], key: str) -> Dict[str, Any]:
    idx = sum(ord(ch) for ch in key) % len(templates)
    return templates[idx]


# --------- Detection helpers ---------
def load_traj_meta(traj_dir: Path) -> TrajMeta:
    traj_path = traj_dir / "traj.json"
    data = json.loads(traj_path.read_text())
    actions_step: List[List[str]] = data["actions_step"]
    action_substeps = int(data.get("traj_params", {}).get("action_substeps", 1))
    frames = data["transforms"]["frames"]
    frame_paths: List[Path] = []
    for f in frames:
        rel = f["file_path"].lstrip("./")
        frame_paths.append(traj_dir / rel)
    return TrajMeta(
        traj_id=traj_dir.name.replace("traj_", ""),
        actions_step=actions_step,
        action_substeps=action_substeps,
        frame_paths=frame_paths,
    )


def relevant_indices(traj: TrajMeta) -> List[int]:
    if traj.traj_id.startswith("T4"):
        candidates = [0, 3, 6, 9]
        return [i for i in candidates if i < len(traj.frame_paths)]
    return list(range(len(traj.frame_paths)))


def load_cached_detections(cache_path: Path) -> Dict[int, List[Detection]]:
    if not cache_path.exists():
        return {}
    frame_map: Dict[int, List[Detection]] = {}
    for line in cache_path.read_text().splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        dets = []
        w_raw = item.get("width")
        h_raw = item.get("height")
        width = int(w_raw) if w_raw is not None else 0
        height = int(h_raw) if h_raw is not None else 0
        for d in item.get("detections", []):
            dets.append(
                Detection(
                    frame_idx=item["frame_idx"],
                    track_id=d.get("track_id"),
                    label_id=d.get("label_id", -1),
                    label=d.get("label", str(d.get("label_id", ""))),
                    bbox=tuple(d["bbox"]),
                    score=float(d["score"]),
                    width=width,
                    height=height,
                )
            )
        frame_map[item["frame_idx"]] = dets
    return frame_map


def _filter_det(
    det: Detection,
    min_area: float,
    max_area: float,
    edge_margin: float,
    conf_keep: float,
) -> bool:
    if det.width <= 0 or det.height <= 0:
        return False
    if det.score < conf_keep:
        return False
    x1, y1, x2, y2 = det.bbox
    if x2 <= x1 or y2 <= y1:
        return False
    area_ratio = (x2 - x1) * (y2 - y1) / float(det.width * det.height)
    if not (min_area <= area_ratio <= max_area):
        return False
    if (
        x1 < edge_margin * det.width
        or y1 < edge_margin * det.height
        or x2 > (1 - edge_margin) * det.width
        or y2 > (1 - edge_margin) * det.height
    ):
        return False
    return True


def _format_bbox(det: Detection) -> str:
    return "[" + ", ".join(str(v) for v in det.bbox_norm_1000()) + "]"


def _visibility_choice(yn: str) -> str:
    return "A" if yn == "yes" else "B"


def _visibility_nl(yn: str, label: str) -> str:
    if yn == "yes":
        return f"The {label} disappears from view."
    return f"The {label} remains visible."


def _identity_choice(yn: str) -> str:
    return "A" if yn == "yes" else "B"


def _identity_nl(yn: str) -> str:
    if yn == "yes":
        return "They are the same physical object instance."
    return "They are different object instances."


def run_detection(
    model_path: str,
    device: Optional[str],
    frame_paths: List[Path],
    frame_indices: List[int],
    conf: float,
    iou: float,
) -> Dict[int, List[Detection]]:
    try:
        from ultralytics import YOLO
    except Exception as exc:
        raise RuntimeError("ultralytics is required for detection. Install with `pip install ultralytics`.") from exc

    model = YOLO(model_path)
    results = model.track(
        source=[str(p) for p in frame_paths],
        conf=conf,
        iou=iou,
        device=device,
        tracker="bytetrack.yaml",
        stream=True,
        verbose=False,
    )

    frame_map: Dict[int, List[Detection]] = {}
    for local_idx, (path, res) in enumerate(zip(frame_paths, results)):
        frame_idx = frame_indices[local_idx]
        with Image.open(path) as img:
            width, height = img.size
        names = getattr(res, "names", {}) or {}
        dets: List[Detection] = []
        ids = res.boxes.id if hasattr(res.boxes, "id") else None
        for j, box in enumerate(res.boxes):
            xyxy = box.xyxy[0].tolist()
            cls_id = int(box.cls[0].item())
            label = names.get(cls_id, str(cls_id))
            track_id = None
            if ids is not None and len(ids) > j and ids[j] is not None:
                try:
                    track_id = int(ids[j].item())
                except Exception:
                    track_id = None
            dets.append(
                Detection(
                    frame_idx=frame_idx,
                    track_id=track_id,
                    label_id=cls_id,
                    label=label,
                    bbox=tuple(float(x) for x in xyxy),
                    score=float(box.conf[0].item()),
                    width=width,
                    height=height,
                )
            )
        frame_map[frame_idx] = dets
    return frame_map


def save_detection_cache(cache_path: Path, frame_map: Dict[int, List[Detection]]) -> None:
    lines = []
    for idx in sorted(frame_map):
        dets = frame_map[idx]
        payload = {
            "frame_idx": idx,
            "width": dets[0].width if dets else None,
            "height": dets[0].height if dets else None,
            "detections": [
                {
                    "track_id": d.track_id,
                    "label_id": d.label_id,
                    "label": d.label,
                    "bbox": list(d.bbox),
                    "bbox_norm": list(d.bbox_norm_1000()),
                    "score": d.score,
                }
                for d in dets
            ],
        }
        lines.append(json.dumps(payload, ensure_ascii=True))
    cache_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


# --------- Sampling logic ---------
def _find_det(
    dets: List[Detection],
    track_id: Optional[int],
    label: str,
    min_area: float,
    max_area: float,
    edge_margin: float,
    conf_keep: float,
) -> Optional[Detection]:
    best: Optional[Detection] = None
    best_score = -math.inf
    for d in dets:
        if track_id is not None and d.track_id != track_id:
            continue
        if track_id is None and label and d.label != label:
            continue
        if not _filter_det(d, min_area, max_area, edge_margin, conf_keep):
            continue
        if d.score > best_score:
            best_score = d.score
            best = d
    return best


def _fill_template(template: Dict[str, Any], mapping: Dict[str, str], images: List[str]) -> Dict[str, Any]:
    obj = json.loads(json.dumps(template))
    for msg in obj["messages"]:
        text = msg["content"]
        for k, v in mapping.items():
            text = text.replace(f"{{{k}}}", v)
        msg["content"] = text
    obj["images"] = images
    return obj


def _find_best_track_pair(
    start_dets: List[Detection],
    target_dets: List[Detection],
    min_area: float,
    max_area: float,
    edge_margin: float,
    conf_keep: float,
) -> Optional[Tuple[Detection, Detection]]:
    best_pair: Optional[Tuple[Detection, Detection]] = None
    best_score = -math.inf
    for start_det in start_dets:
        if start_det.track_id is None:
            continue
        if not _filter_det(start_det, min_area, max_area, edge_margin, conf_keep):
            continue
        partner = _find_det(
            target_dets,
            start_det.track_id,
            start_det.label,
            min_area,
            max_area,
            edge_margin,
            conf_keep,
        )
        if partner is None:
            continue
        score = start_det.score + partner.score
        if score > best_score:
            best_score = score
            best_pair = (start_det, partner)
    return best_pair


def build_d1(
    traj: TrajMeta,
    frame_map: Dict[int, List[Detection]],
    min_area: float,
    max_area: float,
    edge_margin: float,
    conf_keep: float,
) -> Optional[Dict[str, Any]]:
    if not frame_map:
        return None
    target_idx = max(frame_map.keys())
    if target_idx <= 0:
        return None
    pair = _find_best_track_pair(
        frame_map.get(0, []),
        frame_map.get(target_idx, []),
        min_area,
        max_area,
        edge_margin,
        conf_keep,
    )
    if pair is None:
        return None
    start_det, partner = pair
    mapping = {
        "OBJECT": start_det.label,
        "BBOX_1": _format_bbox(start_det),
        "BBOX_2": _format_bbox(partner),
        "ACTION_SEQ": actions_to_text(actions_between(traj.actions_step, 0, target_idx)),
    }
    images = [str(traj.frame_paths[0]), str(traj.frame_paths[target_idx])]
    template = _pick_template(D1_TEMPLATES, f"{traj.traj_id}:{start_det.label}:{target_idx}")
    return _fill_template(template, mapping, images)


def build_d2(
    traj: TrajMeta,
    frame_map: Dict[int, List[Detection]],
    min_area: float,
    max_area: float,
    edge_margin: float,
    conf_keep: float,
) -> Optional[Dict[str, Any]]:
    if not frame_map:
        return None
    start_idx = 0
    target_idx = max(frame_map.keys())
    start_dets = frame_map.get(start_idx, [])
    if not start_dets:
        return None
    start_det = _find_det(start_dets, None, "", min_area, max_area, edge_margin, conf_keep)
    if start_det is None or start_det.track_id is None:
        return None
    actions = actions_between(traj.actions_step, start_idx, target_idx)
    action_text = actions_to_text(actions)
    target_det = _find_det(
        frame_map.get(target_idx, []),
        start_det.track_id,
        start_det.label,
        min_area,
        max_area,
        edge_margin,
        conf_keep,
    )
    yn = "no" if target_det else "yes"
    mapping = {
        "OBJECT": start_det.label,
        "BBOX_1": _format_bbox(start_det),
        "ACTION_SEQ": action_text,
        "YN": yn,
        "VIS_CHOICE": _visibility_choice(yn),
        "VIS_NL": _visibility_nl(yn, start_det.label),
    }
    images = [str(traj.frame_paths[start_idx])]
    template = _pick_template(D2_TEMPLATES, f"{traj.traj_id}:{start_det.label}:{target_idx}")
    return _fill_template(template, mapping, images)


def _pick_target_for_d3(traj: TrajMeta) -> Optional[int]:
    if traj.traj_id.startswith("T4"):
        for candidate in (3, 6):
            if candidate < len(traj.frame_paths):
                return candidate
        return None
    # prefer 2-step gap if available
    if len(traj.frame_paths) > 2:
        return 2
    if len(traj.frame_paths) > 1:
        return 1
    return None


def build_d3(
    traj: TrajMeta,
    frame_map: Dict[int, List[Detection]],
    min_area: float,
    max_area: float,
    edge_margin: float,
    conf_keep: float,
) -> Optional[Dict[str, Any]]:
    target_idx = _pick_target_for_d3(traj)
    if target_idx is None:
        return None
    start_dets = frame_map.get(0, [])
    target_dets = frame_map.get(target_idx, [])
    if not start_dets or not target_dets:
        return None
    pair = _find_best_track_pair(
        start_dets,
        target_dets,
        min_area,
        max_area,
        edge_margin,
        conf_keep,
    )
    if pair is None:
        return None
    start_det, partner = pair
    actions = actions_between(traj.actions_step, 0, target_idx)
    if len(actions) == 0 or len(actions) > 2:
        return None
    mapping = {
        "OBJECT": start_det.label,
        "BBOX_1": _format_bbox(start_det),
        "BBOX_2": _format_bbox(partner),
        "ACTION_SEQ": actions_to_text(actions),
        "ACTION_COUNT": str(len(actions)),
        "ACTION_SEQ_NL": actions_to_natural_text(actions),
    }
    images = [str(traj.frame_paths[0]), str(traj.frame_paths[target_idx])]
    template = _pick_template(D3_TEMPLATES, f"{traj.traj_id}:{start_det.label}:{target_idx}")
    return _fill_template(template, mapping, images)


def build_d4(
    traj: TrajMeta,
    frame_map: Dict[int, List[Detection]],
    min_area: float,
    max_area: float,
    edge_margin: float,
    conf_keep: float,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    if not frame_map:
        return outputs
    target_idx = max(frame_map.keys())
    start_dets = frame_map.get(0, [])
    target_dets = frame_map.get(target_idx, [])
    if not start_dets or not target_dets:
        return outputs
    start_det = _find_det(start_dets, None, "", min_area, max_area, edge_margin, conf_keep)
    if start_det is None:
        return outputs
    actions = actions_between(traj.actions_step, 0, target_idx)
    action_text = actions_to_text(actions)
    # Positive
    pos_partner = _find_det(
        target_dets,
        start_det.track_id,
        start_det.label,
        min_area,
        max_area,
        edge_margin,
        conf_keep,
    )
    if pos_partner:
        mapping = {
            "OBJECT": start_det.label,
            "BBOX_1": _format_bbox(start_det),
            "BBOX_2": _format_bbox(pos_partner),
            "ACTION_SEQ": action_text,
            "YN": "yes",
            "IDENTITY_CHOICE": _identity_choice("yes"),
            "IDENTITY_NL": _identity_nl("yes"),
        }
        template = _pick_template(D4_TEMPLATES, f"{traj.traj_id}:{start_det.label}:{target_idx}:pos")
        outputs.append(_fill_template(template, mapping, [str(traj.frame_paths[0]), str(traj.frame_paths[target_idx])]))

    # Negative
    neg_partner = None
    best_score = -math.inf
    for d in target_dets:
        if d.label != start_det.label:
            continue
        if start_det.track_id is not None and d.track_id == start_det.track_id:
            continue
        if not _filter_det(d, min_area, max_area, edge_margin, conf_keep):
            continue
        if d.score > best_score:
            best_score = d.score
            neg_partner = d
    if neg_partner:
        mapping = {
            "OBJECT": start_det.label,
            "BBOX_1": _format_bbox(start_det),
            "BBOX_2": _format_bbox(neg_partner),
            "ACTION_SEQ": action_text,
            "YN": "no",
            "IDENTITY_CHOICE": _identity_choice("no"),
            "IDENTITY_NL": _identity_nl("no"),
        }
        template = _pick_template(D4_TEMPLATES, f"{traj.traj_id}:{start_det.label}:{target_idx}:neg")
        outputs.append(_fill_template(template, mapping, [str(traj.frame_paths[0]), str(traj.frame_paths[target_idx])]))
    return outputs


# --------- IO ---------
def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def log_skip(traj_dir: Path, exc: Exception) -> None:
    print(f"[skip] {traj_dir}: {type(exc).__name__}: {exc}", file=sys.stderr)


# --------- Main ---------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build D1~D4 detect prompts with cached detection.")
    parser.add_argument("--scenes-root", type=Path, default=Path("svc_data_out/scenes"))
    parser.add_argument("--output", type=Path, default=Path("undetect_data.jsonl"))
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO model path")
    parser.add_argument("--device", type=str, default=None, help="device string for YOLO")
    parser.add_argument("--conf", type=float, default=0.3, help="detection confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="detection NMS IOU threshold")
    parser.add_argument("--min-area", type=float, default=0.01, help="min area ratio filter")
    parser.add_argument("--max-area", type=float, default=0.6, help="max area ratio filter")
    parser.add_argument("--edge-margin", type=float, default=0.01, help="edge margin ratio filter")
    parser.add_argument("--conf-keep", type=float, default=0.3, help="keep threshold for sampling")
    parser.add_argument("--force-detect", action="store_true", help="re-run detection even if cache exists")
    parser.add_argument("--max-scenes", type=int, default=None, help="limit number of scenes")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def process_traj(
    traj_dir: Path,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    traj = load_traj_meta(traj_dir)
    rel_indices = relevant_indices(traj)
    frame_paths = [traj.frame_paths[i] for i in rel_indices]
    cache_path = traj_dir / "detect_result.jsonl"
    if not cache_path.exists() or args.force_detect:
        if args.verbose:
            print(f"[detect] {traj_dir} ({len(frame_paths)} frames)")
        frame_map = run_detection(args.model, args.device, frame_paths, rel_indices, args.conf, args.iou)
        save_detection_cache(cache_path, frame_map)
    else:
        frame_map = load_cached_detections(cache_path)
        if args.verbose:
            print(f"[detect] use cache {cache_path}")

    samples: List[Dict[str, Any]] = []
    d1 = build_d1(traj, frame_map, args.min_area, args.max_area, args.edge_margin, args.conf_keep)
    if d1:
        samples.append(d1)
    d2 = build_d2(traj, frame_map, args.min_area, args.max_area, args.edge_margin, args.conf_keep)
    if d2:
        samples.append(d2)
    d3 = build_d3(traj, frame_map, args.min_area, args.max_area, args.edge_margin, args.conf_keep)
    if d3:
        samples.append(d3)
    d4_list = build_d4(traj, frame_map, args.min_area, args.max_area, args.edge_margin, args.conf_keep)
    samples.extend(d4_list)
    return samples


def main() -> None:
    args = parse_args()
    scenes = sorted([p for p in args.scenes_root.iterdir() if p.is_dir()])
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    total_records = 0
    skipped_trajs = 0
    for scene in scenes:
        traj_dirs = sorted([p for p in scene.iterdir() if p.is_dir() and p.name.startswith("traj_")])
        for traj_dir in traj_dirs:
            try:
                records = process_traj(traj_dir, args)
            except Exception as exc:
                skipped_trajs += 1
                log_skip(traj_dir, exc)
                continue
            append_jsonl(args.output, records)
            total_records += len(records)
    if args.verbose:
        print(f"Wrote {total_records} records to {args.output} (skipped {skipped_trajs} trajectories)")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - simple CLI guard
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
