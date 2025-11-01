"""Continuity helpers for carryover between shots."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np

from vidforge.utils.io import ensure_dir


def save_frame_preview(frame: np.ndarray, out_dir: Path, tag: str) -> Path:
    import cv2

    ensure_dir(out_dir)
    path = out_dir / f"{tag}.png"
    cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    return path


def carryover_frame(registry: Dict[str, np.ndarray], shot_id: str) -> Optional[np.ndarray]:
    return registry.get(shot_id)


def register_last_frame(registry: Dict[str, np.ndarray], shot_id: str, frames: list[np.ndarray]) -> None:
    if not frames:
        return
    registry[shot_id] = frames[-1]
