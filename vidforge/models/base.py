"""Model abstraction layer."""
from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np


@dataclass
class GenerationRequest:
    prompt: str
    seconds: float
    fps: int
    width: int
    height: int
    seed: int
    guidance_scale: float = 7.0
    num_steps: int = 30
    scheduler: str = "ddim"
    precision: str = "fp16"
    enable_guidance: bool = False
    guidance_payload: Optional[dict] = None

    @property
    def num_frames(self) -> int:
        return max(1, int(round(self.seconds * self.fps)))


class BaseVideoModel(abc.ABC):
    """Shared interface for all diffusion backends."""

    name: str

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._loaded = False

    @abc.abstractmethod
    def load(self, fp16: bool = True, bf16: bool = False, **kwargs: dict) -> None:
        """Load model weights into memory."""

    @abc.abstractmethod
    def generate_t2v(self, request: GenerationRequest) -> List[np.ndarray]:
        """Return a list of frames in RGB uint8 layout."""

    @abc.abstractmethod
    def generate_i2v(self, init_frame: np.ndarray, request: GenerationRequest) -> List[np.ndarray]:
        """Render continuation frames given an initial RGB frame."""

    def ensure_loaded(self, **kwargs: dict) -> None:
        if not self._loaded:
            self.load(**kwargs)
            self._loaded = True

    def cleanup(self) -> None:  # pragma: no cover - backend dependent
        pass


class GuidanceAdapter(abc.ABC):
    """Optional style or semantic guidance components."""

    @abc.abstractmethod
    def encode_image(self, path: str) -> dict:
        raise NotImplementedError

    @abc.abstractmethod
    def encode_video(self, path: str) -> dict:
        raise NotImplementedError

    def batch_encode(self, assets: Iterable[str]) -> List[dict]:
        return [self.encode_image(asset) for asset in assets]
