"""Video-as-prompt style guidance adapter."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict

import numpy as np
import cv2

from vidforge.models.base import GuidanceAdapter

logger = logging.getLogger(__name__)


class IdentityGuidance(GuidanceAdapter):
    """Fallback adapter that returns dummy statistics."""

    def encode_image(self, path: str) -> Dict[str, np.ndarray]:
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Failed to load guidance image: {path}")
        feature = image.mean(axis=(0, 1))
        return {"type": "image", "feature": feature.astype(np.float32)}

    def encode_video(self, path: str) -> Dict[str, np.ndarray]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open guidance video: {path}")
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise ValueError("Guidance video contains no frames")
        feature = np.mean([frame.mean(axis=(0, 1)) for frame in frames], axis=0)
        return {"type": "video", "feature": feature.astype(np.float32)}


def load_guidance_adapter(style_path: str | None, video_path: str | None) -> GuidanceAdapter | None:
    if not style_path and not video_path:
        return None
    # TODO: Replace IdentityGuidance with actual VAP encoder integration.
    adapter = IdentityGuidance()
    assets = []
    if style_path:
        assets.append(style_path)
    if video_path:
        assets.append(video_path)
    for asset in assets:
        if not Path(asset).exists():
            logger.warning("Guidance asset missing: %s", asset)
    return adapter
