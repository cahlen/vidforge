"""Video encoding helpers built on ffmpeg-python."""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - import for type checkers only
    import ffmpeg as ffmpeg_module

try:
    import ffmpeg  # type: ignore[import]
except ImportError:  # pragma: no cover - optional dependency
    ffmpeg = None

logger = logging.getLogger(__name__)


def _require_ffmpeg() -> None:
    if ffmpeg is None:
        raise RuntimeError("ffmpeg-python is not installed. Please run `pip install ffmpeg-python`." )


def encode_frames_to_video(
    frames: Iterable[np.ndarray],
    fps: int,
    output_path: Path,
    crf: int = 18,
    codec: str = "libx264",
    pixel_format: str = "yuv420p",
) -> None:
    """Encode an iterable of HxWxC uint8 frames into a video file."""

    _require_ffmpeg()
    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("Cannot encode zero frames") from None

    height, width = first.shape[:2]
    process = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{width}x{height}", framerate=fps)
        .output(
            str(output_path),
            pix_fmt=pixel_format,
            vcodec=codec,
            crf=crf,
            movflags="faststart",
        )
        .overwrite_output()
        .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True)
    )

    process.stdin.write(first.astype(np.uint8).tobytes())
    for frame in iterator:
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    stdout, stderr = process.communicate()
    if process.returncode != 0:  # pragma: no cover - captured via stderr logging
        raise RuntimeError(f"ffmpeg failed: {stderr.decode('utf-8', errors='replace')}")


def concatenate_videos(inputs: List[Path], output: Path, use_crossfade: bool = False, crossfade_seconds: float = 0.5) -> None:
    """Concatenate MP4 clips with optional crossfade transitions."""

    if use_crossfade:
        _concat_with_crossfade(inputs, output, crossfade_seconds)
        return
    _require_ffmpeg()
    list_file = output.parent / "concat_list.txt"
    list_file.write_text("\n".join(f"file '{path}'" for path in inputs) + "\n", encoding="utf-8")
    (
        ffmpeg
        .input(str(list_file), format="concat", safe=0)
        .output(str(output), c="copy")
        .overwrite_output()
        .run()
    )
    list_file.unlink(missing_ok=True)


def _concat_with_crossfade(inputs: List[Path], output: Path, duration: float) -> None:
    logger.warning("Crossfade concatenation not fully implemented; using straight cuts.")
    concatenate_videos(inputs, output, use_crossfade=False)


def mux_audio(video: Path, audio: Path, output: Path) -> None:
    _require_ffmpeg()
    (
        ffmpeg
        .input(str(video))
        .input(str(audio))
        .output(str(output), c_v="copy", c_a="aac", shortest=None)
        .overwrite_output()
        .run()
    )


def extract_last_frame(video: Path, output_image: Path) -> None:
    _require_ffmpeg()
    (
        ffmpeg
        .input(str(video), ss="-1")
        .output(str(output_image), vframes=1)
        .overwrite_output()
        .run()
    )


def detect_vram() -> Optional[int]:
    """Return available GPU memory in MB if nvidia-smi is present."""

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    for line in result.stdout.splitlines():
        line = line.strip().split()[0]
        if line.isdigit():
            return int(line)
    return None
