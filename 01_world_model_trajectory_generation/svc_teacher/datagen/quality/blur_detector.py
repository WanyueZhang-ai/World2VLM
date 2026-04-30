from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class BlurResult:
    score: float
    is_blurry: bool


@dataclass
class QualityResult:
    laplacian_var: float
    brisque_score: float | None
    is_blurry: bool
    is_low_quality: bool


class BlurDetector(ABC):
    @abstractmethod
    def detect(self, image_path: str) -> BlurResult:
        """Return blur score and blur decision."""


class PlaceholderBlurDetector(BlurDetector):
    """Placeholder implementation before a real blur model is selected."""

    def detect(self, image_path: str) -> BlurResult:
        _ = image_path
        return BlurResult(score=0.0, is_blurry=False)


def _laplacian_var_fallback(gray: np.ndarray) -> float:
    # Fallback when cv2 is unavailable.
    lap = (
        np.roll(gray, 1, axis=0)
        + np.roll(gray, -1, axis=0)
        + np.roll(gray, 1, axis=1)
        + np.roll(gray, -1, axis=1)
        - 4.0 * gray
    )
    return float(np.var(lap))


class LaplacianBlurDetector(BlurDetector):
    def __init__(self, threshold: float = 100.0) -> None:
        self.threshold = float(threshold)

    def detect(self, image_path: str) -> BlurResult:
        try:
            import cv2  # type: ignore

            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        except Exception:
            gray = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
            score = _laplacian_var_fallback(gray)
        return BlurResult(score=score, is_blurry=score < self.threshold)


class QualityAssessor:
    def __init__(
        self,
        laplacian_threshold: float = 100.0,
        brisque_threshold: float = 50.0,
        brisque_model_path: str = "",
        brisque_range_path: str = "",
    ) -> None:
        self.laplacian_threshold = float(laplacian_threshold)
        self.brisque_threshold = float(brisque_threshold)
        self.brisque_model_path = str(brisque_model_path).strip()
        self.brisque_range_path = str(brisque_range_path).strip()

    def _compute_brisque(self, bgr_image: Any) -> float | None:
        if not self.brisque_model_path or not self.brisque_range_path:
            return None
        model_path = Path(self.brisque_model_path)
        range_path = Path(self.brisque_range_path)
        if not model_path.exists() or not range_path.exists():
            return None
        try:
            import cv2  # type: ignore

            if not hasattr(cv2, "quality"):
                return None
            score = cv2.quality.QualityBRISQUE_compute(  # type: ignore[attr-defined]
                bgr_image,
                str(model_path),
                str(range_path),
            )
            arr = np.asarray(score).reshape(-1)
            return float(arr[0]) if arr.size > 0 else None
        except Exception:
            return None

    def assess(self, image_path: str) -> QualityResult:
        lap_score = 0.0
        brisque_score: float | None = None
        try:
            import cv2  # type: ignore

            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if gray is None or bgr is None:
                raise RuntimeError(f"Failed to read image: {image_path}")
            lap_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            brisque_score = self._compute_brisque(bgr)
        except Exception:
            gray_np = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
            lap_score = _laplacian_var_fallback(gray_np)

        is_blurry = lap_score < self.laplacian_threshold
        if brisque_score is None:
            is_low_quality = is_blurry
        else:
            is_low_quality = is_blurry or (brisque_score > self.brisque_threshold)
        return QualityResult(
            laplacian_var=lap_score,
            brisque_score=brisque_score,
            is_blurry=is_blurry,
            is_low_quality=is_low_quality,
        )

