"""CogVideoX model wrapper."""
from __future__ import annotations

import logging
from typing import Optional, TYPE_CHECKING, Any

import numpy as np

from vidforge.models.base import BaseVideoModel, GenerationRequest

logger = logging.getLogger(__name__)

if TYPE_CHECKING:  # pragma: no cover - type checking only
    from diffusers import CogVideoXPipeline as CogVideoXPipelineType

try:  # pragma: no cover - optional dependency path
    from diffusers import CogVideoXPipeline  # type: ignore[import]
except ImportError:  # pragma: no cover - executed when dependency missing
    CogVideoXPipeline = None


class CogVideoXModel(BaseVideoModel):
    name = "cogvideox"

    def __init__(self, device: str = "cuda") -> None:
        super().__init__(device=device)
        self.pipeline: Optional[Any] = None

    def load(self, fp16: bool = True, bf16: bool = False, **kwargs: dict) -> None:
        if CogVideoXPipeline is None:
            logger.warning("diffusers with CogVideoXPipeline not available; falling back to placeholder output.")
            return
        try:
            import torch  # type: ignore
        except ImportError:
            logger.warning("PyTorch not installed; using placeholder generator.")
            return
        precision = "fp16" if fp16 else "bf16" if bf16 else "fp32"
        logger.info("Loading CogVideoX pipeline (%s) on %s", precision, self.device)
        # TODO: integrate actual CogVideoX weights download and local path override.
        torch_dtype = torch.float16 if fp16 else torch.bfloat16 if bf16 else torch.float32
        self.pipeline = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-5B", torch_dtype=torch_dtype
        )
        self.pipeline = self.pipeline.to(self.device)

    def generate_t2v(self, request: GenerationRequest) -> list[np.ndarray]:
        self.ensure_loaded()
        if self.pipeline is None:
            return self._placeholder_video(request)
        generator = None
        try:
            import torch  # type: ignore

            generator = torch.Generator(device=self.device)
            generator.manual_seed(request.seed)
        except ImportError:
            logger.debug("Torch generator unavailable; stochastic output may vary.")
        output = self.pipeline(
            prompt=request.prompt,
            num_frames=request.num_frames,
            num_inference_steps=request.num_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
            generator=generator,
        )
        frames = getattr(output, "frames", None)
        if frames is None:
            raise RuntimeError("CogVideoX pipeline returned no frames")
        return [np.clip(frame * 255, 0, 255).astype(np.uint8) for frame in frames]

    def generate_i2v(self, init_frame: np.ndarray, request: GenerationRequest) -> list[np.ndarray]:
        self.ensure_loaded()
        if self.pipeline is None:
            return self._placeholder_video(request, init_frame=init_frame)
        output = self.pipeline(
            prompt=request.prompt,
            image=init_frame,
            num_frames=request.num_frames,
            num_inference_steps=request.num_steps,
            guidance_scale=request.guidance_scale,
            width=request.width,
            height=request.height,
        )
        frames = getattr(output, "frames", None)
        if frames is None:
            raise RuntimeError("CogVideoX pipeline returned no frames")
        return [np.clip(frame * 255, 0, 255).astype(np.uint8) for frame in frames]

    def _placeholder_video(self, request: GenerationRequest, init_frame: Optional[np.ndarray] = None) -> list[np.ndarray]:
        rng = np.random.default_rng(seed=request.seed)
        base = init_frame.astype(np.float32) if init_frame is not None else None
        frames: list[np.ndarray] = []
        frame_count = request.num_frames
        for idx in range(frame_count):
            noise = rng.normal(0, 25, size=(request.height, request.width, 3))
            blend = idx / max(frame_count - 1, 1)
            gradient = np.linspace(0, 255, request.width, dtype=np.float32)
            gradient = np.tile(gradient, (request.height, 1))
            gradient = np.stack([gradient] * 3, axis=-1)
            frame = gradient * (1.0 - blend) + noise
            if base is not None:
                frame = 0.6 * base + 0.4 * frame
            frames.append(np.clip(frame, 0, 255).astype(np.uint8))
        return frames
