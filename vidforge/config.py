"""Global configuration helpers for VidForge."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseSettings, Field


def _default_cache_dir() -> Path:
    return Path.home() / ".cache" / "vidforge"


def _default_models_dir() -> Path:
    return _default_cache_dir() / "models"


class VidForgeSettings(BaseSettings):
    """Runtime configuration resolved from environment variables."""

    cache_dir: Path = Field(default_factory=_default_cache_dir)
    models_dir: Path = Field(default_factory=_default_models_dir)
    ffmpeg_bin: Optional[Path] = None
    enable_rife: bool = True
    enable_esrgan: bool = True

    class Config:
        env_prefix = "VIDFORGE_"
        env_file = ".env"


settings = VidForgeSettings()
