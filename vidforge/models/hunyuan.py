"""HunyuanVideo model wrapper."""
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING, Any

import numpy as np

from vidforge.models.base import BaseVideoModel, GenerationRequest

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover
    from diffusers import HunyuanVideoPipeline as HunyuanPipelineType  # type: ignore[import]

try:  # pragma: no cover - optional dependency path
    from diffusers import HunyuanVideoPipeline  # type: ignore[import]
except ImportError:  # pragma: no cover
    HunyuanVideoPipeline = None


class HunyuanVideoModel(BaseVideoModel):
    name = "hunyuan"

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device=device)
        self.pipeline: Optional[Any] = None

    def load(self, fp16: bool = True, bf16: bool = False, **kwargs: dict) -> None:
        if HunyuanVideoPipeline is None:
            logger.warning("HunyuanVideo pipeline unavailable; falling back to placeholder frames.")
            return
        try:
            import torch  # type: ignore
        except ImportError:
            logger.warning("PyTorch not installed; using placeholder generator.")
            return
        precision = "fp16" if fp16 else "bf16" if bf16 else "fp32"
        logger.info("Loading HunyuanVideo pipeline (%s) on %s", precision, self.device)
        # TODO: integrate official HunyuanVideo loader with local checkpoint paths.
        kwargs = kwargs or {}
        dtype = kwargs.get("torch_dtype")
        if dtype is None:
            dtype = torch.float16 if fp16 else torch.bfloat16 if bf16 else torch.float32
        try:
            self.pipeline = HunyuanVideoPipeline.from_pretrained(
                "TencentARC/HunyuanVideo", torch_dtype=dtype
            )
            self.pipeline = self.pipeline.to(self.device)
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.warning("Failed to load HunyuanVideo pipeline (%s); using placeholder frames.", exc)
            self.pipeline = None

    def generate_t2v(self, request: GenerationRequest) -> list[np.ndarray]:
        self.ensure_loaded()
        if self.pipeline is None:
            return self._placeholder_video(request)
        video = self.pipeline(
            prompt=request.prompt,
            num_frames=request.num_frames,
            num_inference_steps=request.num_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
        )
        frames = getattr(video, "frames", None)
        if frames is None:
            raise RuntimeError("Hunyuan pipeline returned no frames")
        return [np.clip(frame * 255, 0, 255).astype(np.uint8) for frame in frames]

    def generate_i2v(self, init_frame: np.ndarray, request: GenerationRequest) -> list[np.ndarray]:
        self.ensure_loaded()
        if self.pipeline is None:
            return self._placeholder_video(request, init_frame=init_frame)
        video = self.pipeline(
            prompt=request.prompt,
            video=init_frame,
            num_frames=request.num_frames,
            num_inference_steps=request.num_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
        )
        frames = getattr(video, "frames", None)
        if frames is None:
            raise RuntimeError("Hunyuan pipeline returned no frames")
        return [np.clip(frame * 255, 0, 255).astype(np.uint8) for frame in frames]

    def _placeholder_video(self, request: GenerationRequest, init_frame: Optional[np.ndarray] = None) -> list[np.ndarray]:
        rng = np.random.default_rng(seed=request.seed)
        base = init_frame.astype(np.float32) if init_frame is not None else None
        frames: list[np.ndarray] = []
        frame_count = request.num_frames
        for idx in range(frame_count):
            noise = rng.uniform(0, 255, size=(request.height, request.width, 3))
            blend = np.sin(idx / max(frame_count, 1) * np.pi)
            frame = noise * blend + (base if base is not None else 127)
            frames.append(np.clip(frame, 0, 255).astype(np.uint8))
        return frames
