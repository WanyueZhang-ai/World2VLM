from __future__ import annotations

from pathlib import Path
from typing import Any


class YOLODetector:
    def __init__(self, model_path: str, device: str | None = None) -> None:
        try:
            from ultralytics import YOLO
        except Exception as exc:
            raise RuntimeError("ultralytics is required for YOLODetector") from exc
        self.model = YOLO(model_path)
        self.device = device

    def detect(self, image_path: Path, conf: float = 0.4) -> list[dict[str, Any]]:
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            verbose=False,
            device=self.device,
        )
        if not results:
            return []
        dets = []
        for box in results[0].boxes:
            xyxy = box.xyxy[0].tolist()
            dets.append(
                {
                    "xyxy": tuple(float(x) for x in xyxy),
                    "conf": float(box.conf[0].item()),
                    "cls": int(box.cls[0].item()),
                }
            )
        return dets
