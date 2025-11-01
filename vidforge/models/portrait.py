"""Portrait generation utilities backed by Stable Diffusion or Flux.

The loader is optional: if Stable Diffusion cannot be loaded (missing
dependencies, missing weights, or GPU constraints), a deterministic
placeholder portrait is produced instead so the downstream pipeline can
continue operating in degraded mode. This keeps unit tests light-weight
and gives users clear logging about missing assets.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

try:  # pragma: no cover - heavy optional dependency
    from diffusers import StableDiffusionPipeline  # type: ignore[import]
except ImportError:  # pragma: no cover
    StableDiffusionPipeline = None  # type: ignore[misc]

try:  # pragma: no cover - heavy optional dependency
    from diffusers import FluxPipeline  # type: ignore[import]
except ImportError:  # pragma: no cover
    FluxPipeline = None  # type: ignore[misc]


@dataclass(slots=True)
class PortraitRequest:
    prompt: str
    negative_prompt: Optional[str] = None
    width: int = 512
    height: int = 512
    num_inference_steps: int = 30
    guidance_scale: float = 7.5
    seed: int = 343455122
    use_model: bool = True


class PortraitGenerator:
    """Generate photographic portraits from text prompts."""

    MODEL_ENV = "VIDFORGE_PORTRAIT_MODEL_ID"
    DEFAULT_MODEL_ID = "runwayml/stable-diffusion-v1-5"

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.pipeline = None
        self._pipeline_kind: Optional[str] = None

    def load(self, *, fp16: bool = True, disable: bool = False) -> None:
        """Lazy-load the configured diffusion pipeline if available."""

        if disable or os.environ.get("VIDFORGE_DISABLE_PORTRAITS") == "1":
            logger.info("Portrait generation disabled via environment flag.")
            self.pipeline = None
            self._pipeline_kind = None
            return
        model_id = os.environ.get(self.MODEL_ENV, self.DEFAULT_MODEL_ID)
        if StableDiffusionPipeline is None and ("flux" not in model_id.lower()):
            logger.debug("diffusers StableDiffusionPipeline unavailable; will only load Flux models if possible.")
        token = os.environ.get("HF_TOKEN") or os.environ.get("HF_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        try:
            import torch  # type: ignore[import]
        except ImportError:  # pragma: no cover
            logger.warning("PyTorch not installed; portrait generation will use placeholders.")
            self.pipeline = None
            self._pipeline_kind = None
            return
        is_flux_model = "flux" in model_id.lower()
        pipeline_cls = FluxPipeline if is_flux_model else StableDiffusionPipeline
        if pipeline_cls is None:
            logger.warning("Requested portrait model '%s' is unavailable; using placeholder portraits.", model_id)
            self.pipeline = None
            self._pipeline_kind = None
            return

        def _supports_bf16() -> bool:
            cuda_module = getattr(torch, "cuda", None)
            if cuda_module is None:
                return False
            if not getattr(cuda_module, "is_available", lambda: False)():
                return False
            supports_attr = getattr(cuda_module, "is_bf16_supported", None)
            try:
                if callable(supports_attr):
                    return bool(supports_attr())
            except Exception:  # pragma: no cover - defensive fallback
                return False
            return False

        if is_flux_model and fp16 and _supports_bf16():
            torch_dtype = torch.bfloat16
        elif fp16:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        logger.info("Loading portrait diffusion pipeline '%s' (dtype=%s)", model_id, torch_dtype)
        try:
            load_kwargs = {"torch_dtype": torch_dtype}
            if token:
                load_kwargs["token"] = token
            if is_flux_model:
                pipeline = pipeline_cls.from_pretrained(  # type: ignore[misc,call-arg]
                    model_id,
                    **load_kwargs,
                )
                if hasattr(pipeline, "enable_model_cpu_offload"):
                    try:
                        pipeline.enable_model_cpu_offload()
                    except Exception:  # pragma: no cover - optional optimisation
                        logger.debug("Flux CPU offload unavailable; pipeline will run on %s.", self.device)
                        pipeline = pipeline.to(self.device)
                else:
                    pipeline = pipeline.to(self.device)
                vae = getattr(pipeline, "vae", None)
                if vae is not None:
                    if hasattr(vae, "enable_slicing"):
                        try:
                            vae.enable_slicing()
                        except Exception:
                            logger.debug("Flux VAE slicing unavailable.")
                    if hasattr(vae, "enable_tiling"):
                        try:
                            vae.enable_tiling()
                        except Exception:
                            logger.debug("Flux VAE tiling unavailable.")
            else:
                pipeline = pipeline_cls.from_pretrained(  # type: ignore[misc,call-arg]
                    model_id,
                    safety_checker=None,
                    **load_kwargs,
                )
                pipeline = pipeline.to(self.device)
                if hasattr(pipeline, "enable_xformers_memory_efficient_attention"):
                    try:
                        pipeline.enable_xformers_memory_efficient_attention()  # type: ignore[attr-defined]
                    except Exception:  # pragma: no cover - optional optimisation
                        logger.debug("xFormers memory optimisation unavailable; continuing without it.")
            self.pipeline = pipeline
            self._pipeline_kind = "flux" if is_flux_model else "sd"
            if is_flux_model and torch_dtype == torch.float16:
                logger.info("Flux pipeline loaded in FP16; consider enabling BF16 for best quality if supported.")
        except Exception as exc:  # pragma: no cover - runtime fetch failure
            logger.warning(
                "Failed to load portrait diffusion pipeline '%s' (%s); using placeholder portraits.",
                model_id,
                exc,
            )
            self.pipeline = None
            self._pipeline_kind = None

    def generate(self, request: PortraitRequest) -> Image.Image:
        if not request.use_model or self.pipeline is None:
            return self._placeholder(request)
        try:
            import torch  # type: ignore[import]
        except ImportError:  # pragma: no cover
            logger.debug("Torch missing at generation time; falling back to placeholder portrait.")
            return self._placeholder(request)
        pipeline_device = getattr(self.pipeline, "_execution_device", None)
        if pipeline_device is None:
            pipeline_device = getattr(self.pipeline, "device", self.device)
        try:
            pipeline_device_obj = torch.device(pipeline_device)
        except Exception:
            pipeline_device_obj = torch.device(self.device)
        generator_device = (
            f"{pipeline_device_obj.type}:{pipeline_device_obj.index}"
            if pipeline_device_obj.index is not None
            else pipeline_device_obj.type
        )
        generator = torch.Generator(device=generator_device)
        generator.manual_seed(request.seed)
        pipeline_name = "Flux" if self._pipeline_kind == "flux" else "Stable Diffusion"
        logger.info("Generating portrait using %s pipeline (%s steps)", pipeline_name, request.num_inference_steps)
        call_kwargs = {
            "prompt": request.prompt,
            "width": request.width,
            "height": request.height,
            "num_inference_steps": request.num_inference_steps,
            "guidance_scale": request.guidance_scale,
            "generator": generator,
        }
        if self._pipeline_kind == "flux":
            default_guidance = PortraitRequest.__dataclass_fields__["guidance_scale"].default  # type: ignore[index]
            if request.guidance_scale == default_guidance:
                # Flux guidance-distilled checkpoints expect a lower CFG for best fidelity.
                call_kwargs["guidance_scale"] = 3.5
                logger.debug("Flux guidance scale overridden to 3.5 for improved portrait fidelity.")
        if request.negative_prompt:
            call_kwargs["negative_prompt"] = request.negative_prompt
        result = self.pipeline(**call_kwargs)  # type: ignore[operator]
        images = getattr(result, "images", None)
        if not images:
            logger.warning("%s pipeline returned no images; using placeholder portrait instead.", pipeline_name)
            return self._placeholder(request)
        image = images[0]
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.asarray(image, dtype=np.uint8))
        return image.convert("RGB")

    def _placeholder(self, request: PortraitRequest) -> Image.Image:
        """Create a deterministic placeholder portrait."""

        rng = np.random.default_rng(seed=request.seed)
        width, height = request.width, request.height
        base = np.linspace(0, 255, width, dtype=np.float32)
        gradient = np.tile(base, (height, 1))
        gradient = np.stack([gradient, gradient[::-1], gradient], axis=-1)
        noise = rng.normal(0, 32, size=(height, width, 3))
        data = np.clip(gradient + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(data, mode="RGB")
        draw = ImageDraw.Draw(image)
        text = "PORTRAIT\nUNAVAILABLE"
        try:
            font = ImageFont.load_default()
        except Exception:  # pragma: no cover - PIL internal issue
            font = None
        draw.multiline_text((width * 0.05, height * 0.1), text, fill=(255, 255, 255), font=font, spacing=4)
        prompt_preview = request.prompt[:80] + ("…" if len(request.prompt) > 80 else "")
        draw.text((width * 0.05, height * 0.8), prompt_preview, fill=(255, 255, 255), font=font)
        return image

    def save_placeholder(self, path: Path, request: PortraitRequest) -> Image.Image:
        portrait = self._placeholder(request)
        portrait.save(path)
        return portrait
