"""Shot rendering orchestration."""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
from rich.progress import Progress, TimeElapsedColumn, TimeRemainingColumn

from vidforge.models.base import BaseVideoModel, GenerationRequest
from vidforge.models.cogvideox import CogVideoXModel
from vidforge.models.guidance_vap import load_guidance_adapter
from vidforge.models.hunyuan import HunyuanVideoModel
from vidforge.pipeline.continuity import register_last_frame, carryover_frame, save_frame_preview
from vidforge.pipeline.shots import RenderPlan, ShotConfig
from vidforge.utils.io import ensure_dir, log_path
from vidforge.utils.seed import resolve_seed
from vidforge.utils.video import encode_frames_to_video

logger = logging.getLogger(__name__)

MODEL_FACTORY = {
    "cogvideox": CogVideoXModel,
    "hunyuan": HunyuanVideoModel,
}


@dataclass
class RenderOptions:
    out_dir: Path
    model_name: str
    seed: Optional[int]
    default_seconds: Optional[float]
    fp16: bool
    bf16: bool
    guidance_scale: float
    num_steps: int
    scheduler: str
    vap_style: Optional[str]
    vap_video: Optional[str]
    resume: bool
    keep_intermediates: bool


def _instantiate_model(name: str) -> BaseVideoModel:
    try:
        model_cls = MODEL_FACTORY[name]
    except KeyError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Unsupported model: {name}") from exc
    return model_cls()


def _read_last_frame(video_path: Path) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count - 1)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _render_shot(
    model: BaseVideoModel,
    shot: ShotConfig,
    plan: RenderPlan,
    options: RenderOptions,
    continuity_buffer: Dict[str, np.ndarray],
    previous_shot_id: Optional[str],
    guidance_payload: Optional[dict],
) -> Path:
    shot_dir = ensure_dir(options.out_dir / "shots" / shot.id)
    frames_dir = ensure_dir(shot_dir / "frames")
    output_video = shot_dir / f"{shot.id}.mp4"

    if options.resume and output_video.exists():
        logger.info("Skipping shot %s (exists)", shot.id)
        cached = _read_last_frame(output_video)
        if cached is not None:
            register_last_frame(continuity_buffer, shot.id, [cached])
            preview_dir = ensure_dir(options.out_dir / "previews")
            save_frame_preview(cached, preview_dir, f"{shot.id}_resume_last")
        return output_video

    fps = shot.fps or plan.fps
    width = shot.width or plan.width
    height = shot.height or plan.height
    seconds = shot.seconds or options.default_seconds or 6.0

    request = GenerationRequest(
        prompt=shot.prompt,
        seconds=seconds,
        fps=fps,
        width=width,
        height=height,
        seed=resolve_seed(options.seed, shot.prompt, salt=shot.id),
        guidance_scale=options.guidance_scale,
        num_steps=options.num_steps,
        scheduler=options.scheduler,
        precision="fp16" if options.fp16 else "bf16" if options.bf16 else "fp32",
        enable_guidance=guidance_payload is not None,
        guidance_payload=guidance_payload,
    )

    model.ensure_loaded(fp16=options.fp16, bf16=options.bf16)

    if shot.type == "i2v":
        init_frame = None
        if shot.carryover and previous_shot_id:
            init_frame = carryover_frame(continuity_buffer, previous_shot_id)
            if init_frame is None:
                logger.warning("Shot %s requested carryover from %s but buffer was empty", shot.id, previous_shot_id)
        if init_frame is None:
            init_frame = np.zeros((height, width, 3), dtype=np.uint8)
        frames = model.generate_i2v(init_frame, request)
    else:
        frames = model.generate_t2v(request)

    preview_dir = ensure_dir(options.out_dir / "previews")
    save_frame_preview(frames[0], preview_dir, f"{shot.id}_first")
    save_frame_preview(frames[-1], preview_dir, f"{shot.id}_last")
    register_last_frame(continuity_buffer, shot.id, frames)

    for idx, frame in enumerate(frames):
        cv2.imwrite(str(frames_dir / f"{idx:04d}.png"), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    encode_frames_to_video(frames, fps=fps, output_path=output_video)
    log_path(output_video)
    if not options.keep_intermediates:
        shutil.rmtree(frames_dir, ignore_errors=True)

    return output_video


def render_plan(plan: RenderPlan, options: RenderOptions) -> list[Path]:
    ensure_dir(options.out_dir)
    model = _instantiate_model(options.model_name)
    guidance = load_guidance_adapter(options.vap_style, options.vap_video)
    guidance_payload = None
    if guidance is not None:
        if options.vap_style:
            guidance_payload = guidance.encode_image(options.vap_style)
        elif options.vap_video:
            guidance_payload = guidance.encode_video(options.vap_video)

    continuity_buffer: Dict[str, np.ndarray] = {}
    outputs: list[Path] = []
    previous_shot_id: Optional[str] = None

    with Progress(
        "[progress.description]{task.description}",
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Rendering shots", total=len(plan.shots))
        for shot in plan.shots:
            video_path = _render_shot(
                model=model,
                shot=shot,
                plan=plan,
                options=options,
                continuity_buffer=continuity_buffer,
                previous_shot_id=previous_shot_id,
                guidance_payload=guidance_payload,
            )
            outputs.append(video_path)
            previous_shot_id = shot.id
            progress.advance(task)

    return outputs
