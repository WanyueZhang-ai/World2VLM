from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from datagen.motion.action_space import Act, ActionSpec, canonicalize


D2_TEMPLATE = {
    "task_type": "D2",
    "messages": [
        {
            "role": "user",
            "content": "<image>In the image, the {OBJECT} is at bbox {BBOX_1}. After the camera does \"{ACTION_SEQ}\", does this {OBJECT} disappear from view? Answer: yes or no.",
        },
        {"role": "assistant", "content": "{YN}"},
    ],
    "images": ["{IMG_1}"],
}

D3_TEMPLATE = {
    "task_type": "D3",
    "messages": [
        {
            "role": "user",
            "content": "<image><image>The {OBJECT} in the first image (bbox {BBOX_1}) and the {OBJECT} in the second image (bbox {BBOX_2}) are the same physical object. the camera used 1 or 2 actions in order. Write the full action sequence using ';' as a separator. Allowed actions:\n- move forward/backward/left/right X meters\n- turn left/right X degrees",
        },
        {"role": "assistant", "content": "{ACTION_SEQ}"},
    ],
    "images": ["{IMG_1}", "{IMG_2}"],
}

D4_TEMPLATE = {
    "task_type": "D4",
    "messages": [
        {
            "role": "user",
            "content": "<image><image>In the first image, the {OBJECT} is at bbox {BBOX_1}. After the camera does \"{ACTION_SEQ}\", the second image shows a {OBJECT} at bbox {BBOX_2}. Are these the same physical object instance? Answer: yes or no.",
        },
        {"role": "assistant", "content": "{YN}"},
    ],
    "images": ["{IMG_1}", "{IMG_2}"],
}


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

    def bbox_norm(self) -> Tuple[float, float, float, float]:
        x1, y1, x2, y2 = self.bbox
        return (
            x1 / self.width,
            y1 / self.height,
            x2 / self.width,
            y2 / self.height,
        )


@dataclass
class TrajMeta:
    traj_id: str
    actions_step: List[List[str]]
    frame_paths: List[Path]
    key_frame_indices: List[int]


_ACT_SHORT_TO_ENUM = {
    "TL": Act.TURN_LEFT,
    "TR": Act.TURN_RIGHT,
    "FWD": Act.FORWARD,
    "BACK": Act.BACKWARD,
    "LEFT": Act.SHIFT_LEFT,
    "RIGHT": Act.SHIFT_RIGHT,
}

_ACT_TO_TEXT = {
    Act.TURN_LEFT: "turn left",
    Act.TURN_RIGHT: "turn right",
    Act.FORWARD: "move forward",
    Act.BACKWARD: "move backward",
    Act.SHIFT_LEFT: "move left",
    Act.SHIFT_RIGHT: "move right",
}


def _parse_action(text: str) -> ActionSpec:
    if "_" not in text:
        raise ValueError(f"Invalid action string: {text}")
    short, raw_val = text.split("_", 1)
    act = _ACT_SHORT_TO_ENUM[short]
    return ActionSpec(act, float(raw_val))


def _format_value(val: float) -> str:
    if abs(val - round(val)) < 1e-6:
        return str(int(round(val)))
    return f"{val:.6f}".rstrip("0").rstrip(".")


def actions_between(actions_step: List[List[str]], start_idx: int, end_idx: int) -> List[ActionSpec]:
    merged: List[ActionSpec] = []
    for i in range(start_idx + 1, end_idx + 1):
        merged.extend(_parse_action(a) for a in actions_step[i])
    return canonicalize(merged)


def actions_to_text(actions: Sequence[ActionSpec]) -> str:
    parts = []
    for act in actions:
        unit = "degrees" if act.act in {Act.TURN_LEFT, Act.TURN_RIGHT} else "meters"
        parts.append(f"{_ACT_TO_TEXT[act.act]} {_format_value(act.magnitude)} {unit}")
    return "; ".join(parts)


def load_traj_meta(traj_dir: Path) -> TrajMeta:
    traj_path = traj_dir / "traj.json"
    data = json.loads(traj_path.read_text(encoding="utf-8"))
    actions_step: List[List[str]] = data["actions_step"]
    frames = data["transforms"]["frames"]
    frame_paths: List[Path] = []
    for frame in frames:
        rel = frame["file_path"].lstrip("./")
        frame_paths.append(traj_dir / rel)
    key_frame_indices = [int(v) for v in data.get("key_frame_indices", [])]
    if not key_frame_indices:
        key_frame_indices = [int(v) for v in data.get("traj_params", {}).get("key_frame_indices", [])]
    return TrajMeta(
        traj_id=traj_dir.name.replace("traj_", ""),
        actions_step=actions_step,
        frame_paths=frame_paths,
        key_frame_indices=key_frame_indices,
    )


def relevant_indices(traj: TrajMeta) -> List[int]:
    if traj.key_frame_indices:
        idxs = {0}
        idxs.update(i for i in traj.key_frame_indices if 0 <= i < len(traj.frame_paths))
        idxs.add(len(traj.frame_paths) - 1)
        return sorted(idxs)
    return list(range(len(traj.frame_paths)))


def load_cached_detections(cache_path: Path) -> Dict[int, List[Detection]]:
    if not cache_path.exists():
        return {}
    frame_map: Dict[int, List[Detection]] = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        dets = []
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


def _format_bbox(bbox: Tuple[float, float, float, float]) -> str:
    return "[" + ", ".join(f"{v:.1f}" for v in bbox) + "]"


def run_detection(
    model_path: str,
    device: Optional[str],
    frame_paths: List[Path],
    frame_indices: List[int],
    conf: float,
    iou: float,
) -> Dict[int, List[Detection]]:
    from PIL import Image

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
                    "bbox_norm": list(d.bbox_norm()),
                    "score": d.score,
                }
                for d in dets
            ],
        }
        lines.append(json.dumps(payload, ensure_ascii=True))
    cache_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


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
    for det in dets:
        if track_id is not None and det.track_id != track_id:
            continue
        if track_id is None and label and det.label != label:
            continue
        if not _filter_det(det, min_area, max_area, edge_margin, conf_keep):
            continue
        if det.score > best_score:
            best_score = det.score
            best = det
    return best


def _fill_template(template: Dict[str, Any], mapping: Dict[str, str], images: List[str]) -> Dict[str, Any]:
    obj = json.loads(json.dumps(template))
    for msg in obj["messages"]:
        text = msg["content"]
        for key, value in mapping.items():
            text = text.replace(f"{{{key}}}", value)
        msg["content"] = text
    obj["images"] = images
    return obj


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
        "BBOX_1": _format_bbox(start_det.bbox),
        "ACTION_SEQ": action_text,
        "YN": yn,
    }
    return _fill_template(D2_TEMPLATE, mapping, [str(traj.frame_paths[start_idx])])


def _pick_target_for_d3(traj: TrajMeta) -> Optional[int]:
    candidates = [idx for idx in traj.key_frame_indices if idx > 0]
    candidates = sorted(set(candidates))
    for candidate in candidates:
        if candidate < len(traj.frame_paths):
            return candidate
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
    start_det = _find_det(start_dets, None, "", min_area, max_area, edge_margin, conf_keep)
    if start_det is None or start_det.track_id is None:
        return None
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
        return None
    actions = actions_between(traj.actions_step, 0, target_idx)
    if len(actions) == 0 or len(actions) > 2:
        return None
    mapping = {
        "OBJECT": start_det.label,
        "BBOX_1": _format_bbox(start_det.bbox),
        "BBOX_2": _format_bbox(partner.bbox),
        "ACTION_SEQ": actions_to_text(actions),
    }
    return _fill_template(D3_TEMPLATE, mapping, [str(traj.frame_paths[0]), str(traj.frame_paths[target_idx])])


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
            "BBOX_1": _format_bbox(start_det.bbox),
            "BBOX_2": _format_bbox(pos_partner.bbox),
            "ACTION_SEQ": action_text,
            "YN": "yes",
        }
        outputs.append(_fill_template(D4_TEMPLATE, mapping, [str(traj.frame_paths[0]), str(traj.frame_paths[target_idx])]))

    neg_partner = None
    best_score = -math.inf
    for det in target_dets:
        if det.label != start_det.label:
            continue
        if start_det.track_id is not None and det.track_id == start_det.track_id:
            continue
        if not _filter_det(det, min_area, max_area, edge_margin, conf_keep):
            continue
        if det.score > best_score:
            best_score = det.score
            neg_partner = det
    if neg_partner:
        mapping = {
            "OBJECT": start_det.label,
            "BBOX_1": _format_bbox(start_det.bbox),
            "BBOX_2": _format_bbox(neg_partner.bbox),
            "ACTION_SEQ": action_text,
            "YN": "no",
        }
        outputs.append(_fill_template(D4_TEMPLATE, mapping, [str(traj.frame_paths[0]), str(traj.frame_paths[target_idx])]))
    return outputs


def append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    records_list = list(records)
    if not records_list:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for rec in records_list:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build D2~D4 detect prompts with cached detection.")
    parser.add_argument("--scenes-root", type=Path, default=Path("outputs/scenes"))
    parser.add_argument("--output", type=Path, default=Path("outputs/data_detect.jsonl"))
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


def process_traj(traj_dir: Path, args: argparse.Namespace) -> List[Dict[str, Any]]:
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
    d2 = build_d2(traj, frame_map, args.min_area, args.max_area, args.edge_margin, args.conf_keep)
    if d2:
        samples.append(d2)
    d3 = build_d3(traj, frame_map, args.min_area, args.max_area, args.edge_margin, args.conf_keep)
    if d3:
        samples.append(d3)
    samples.extend(build_d4(traj, frame_map, args.min_area, args.max_area, args.edge_margin, args.conf_keep))
    return samples


def main() -> None:
    args = parse_args()
    scenes = sorted([path for path in args.scenes_root.iterdir() if path.is_dir()])
    if args.max_scenes is not None:
        scenes = scenes[: args.max_scenes]
    all_records: List[Dict[str, Any]] = []
    for scene in scenes:
        traj_dirs = sorted([path for path in scene.iterdir() if path.is_dir() and path.name.startswith("traj_")])
        for traj_dir in traj_dirs:
            all_records.extend(process_traj(traj_dir, args))
    append_jsonl(args.output, all_records)
    if args.verbose:
        print(f"Wrote {len(all_records)} records to {args.output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

