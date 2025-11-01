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

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional dependency
    imageio = None  # type: ignore

logger = logging.getLogger(__name__)


def _require_ffmpeg() -> None:
    if ffmpeg is None:
        raise RuntimeError("ffmpeg-python is not installed. Please run `pip install ffmpeg-python`.")


def _encode_with_ffmpeg(
    frames: List[np.ndarray],
    fps: int,
    output_path: Path,
    crf: int,
    codec: str,
    pixel_format: str,
) -> None:
    _require_ffmpeg()
    height, width = frames[0].shape[:2]
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

    try:
        for frame in frames:
            process.stdin.write(frame.astype(np.uint8).tobytes())
    finally:
        process.stdin.close()
    stdout, stderr = process.communicate()
    if process.returncode != 0:  # pragma: no cover - captured via stderr logging
        raise RuntimeError(f"ffmpeg failed: {stderr.decode('utf-8', errors='replace')}")


def _encode_with_imageio(frames: List[np.ndarray], fps: int, output_path: Path) -> None:
    if imageio is None:
        raise RuntimeError("imageio is not installed; cannot encode video using this backend.")
    with imageio.get_writer(
        uri=str(output_path),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for frame in frames:
            writer.append_data(frame)


def _encode_with_opencv(frames: List[np.ndarray], fps: int, output_path: Path) -> None:
    if cv2 is None:
        raise RuntimeError("OpenCV is not installed; cannot fall back when ffmpeg is unavailable.")
    height, width = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV VideoWriter failed to open output path.")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def encode_frames_to_video(
    frames: Iterable[np.ndarray],
    fps: int,
    output_path: Path,
    crf: int = 18,
    codec: str = "libx264",
    pixel_format: str = "yuv420p",
) -> None:
    """Encode an iterable of HxWxC uint8 frames into a video file."""

    iterator = iter(frames)
    try:
        first = next(iterator)
    except StopIteration:
        raise ValueError("Cannot encode zero frames") from None
    first_u8 = first.astype(np.uint8, copy=False)
    frame_buffer = [first_u8]
    for frame in iterator:
        frame_buffer.append(np.asarray(frame, dtype=np.uint8))

    if ffmpeg is not None:
        try:
            _encode_with_ffmpeg(frame_buffer, fps, output_path, crf, codec, pixel_format)
            return
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.warning("ffmpeg encoding failed (%s); trying alternate encoders.", exc)
            output_path.unlink(missing_ok=True)

    if imageio is not None:
        try:
            _encode_with_imageio(frame_buffer, fps, output_path)
            return
        except Exception as exc:  # pragma: no cover - runtime safeguard
            logger.warning("imageio encoding failed (%s); attempting OpenCV fallback.", exc)
            output_path.unlink(missing_ok=True)

    _encode_with_opencv(frame_buffer, fps, output_path)


def concatenate_videos(inputs: List[Path], output: Path, use_crossfade: bool = False, crossfade_seconds: float = 0.5) -> None:
    """Concatenate MP4 clips with optional crossfade transitions."""

    if use_crossfade:
        _concat_with_crossfade(inputs, output, crossfade_seconds)
        return
    _require_ffmpeg()
    list_file = output.parent / "concat_list.txt"
    list_file.write_text(
        "\n".join(f"file '{path.resolve()}'" for path in inputs) + "\n",
        encoding="utf-8",
    )
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
    video_stream = ffmpeg.input(str(video))
    audio_stream = ffmpeg.input(str(audio))
    (
        ffmpeg
        .output(video_stream, audio_stream, str(output), vcodec="copy", acodec="aac", shortest=None)
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
