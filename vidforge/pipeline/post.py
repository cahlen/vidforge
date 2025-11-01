"""Post-processing pipeline: smoothing, upscaling, concatenation, audio."""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import List, Optional

from rich.console import Console

from vidforge.utils.io import ensure_dir
from vidforge.utils.video import concatenate_videos, mux_audio

logger = logging.getLogger(__name__)
console = Console()


def _copy_passthrough(src: Path, dst: Path) -> Path:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def apply_rife(video: Path, out_dir: Path) -> Path:
    """Placeholder RIFE hook; currently copies input."""

    target = out_dir / f"{video.stem}_rife.mp4"
    console.log("[yellow]RIFE not configured; copying input video[/]")
    return _copy_passthrough(video, target)


def apply_esrgan(video: Path, out_dir: Path) -> Path:
    """Placeholder ESRGAN hook; currently copies input."""

    target = out_dir / f"{video.stem}_esrgan.mp4"
    console.log("[yellow]ESRGAN not configured; copying input video[/]")
    return _copy_passthrough(video, target)


def combine_shots(
    shot_videos: List[Path],
    out_dir: Path,
    use_crossfade: bool,
    crossfade_seconds: float,
    apply_rife_opt: bool,
    apply_esrgan_opt: bool,
) -> Path:
    ensure_dir(out_dir)
    processed: List[Path] = []

    for video in shot_videos:
        current = video
        if apply_rife_opt:
            current = apply_rife(current, out_dir)
        if apply_esrgan_opt:
            current = apply_esrgan(current, out_dir)
        processed.append(current)

    final_path = out_dir / "final.mp4"
    concatenate_videos(processed, final_path, use_crossfade, crossfade_seconds)
    return final_path


def mux_audio_if_present(video: Path, audio_path: Optional[str], out_dir: Path) -> Path:
    if not audio_path:
        return video
    audio_file = Path(audio_path)
    if not audio_file.exists():
        logger.warning("Audio file %s missing; skipping mux", audio_file)
        return video
    if audio_file.stat().st_size == 0:
        logger.warning("Audio file %s is empty; skipping mux", audio_file)
        return video
    target = out_dir / "final_with_audio.mp4"
    try:
        mux_audio(video, audio_file, target)
    except Exception as exc:  # pragma: no cover - runtime safeguard
        logger.warning("Failed to mux audio (%s); returning video-only output.", exc)
        return video
    return target
