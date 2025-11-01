"""Shot plan ingestion and validation."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field, field_validator

from vidforge.utils.io import load_json


class ShotConfig(BaseModel):
    id: str
    type: str = Field(pattern="^(t2v|i2v)$")
    seconds: Optional[float] = Field(default=None, gt=0)
    prompt: str
    carryover: bool = False
    fps: Optional[int] = None
    width: Optional[int] = None
    height: Optional[int] = None

    @field_validator("seconds")
    @classmethod
    def validate_duration(cls, value: Optional[float]) -> Optional[float]:
        if value is None:
            return value
        if value > 16:
            raise ValueError("Individual shots must be <= 16 seconds to ensure stability")
        return value


class RenderPlan(BaseModel):
    project_title: str
    fps: int = 24
    width: int = 1280
    height: int = 720
    audio_file: Optional[str] = None
    model: str = Field(default="cogvideox", pattern="^(cogvideox|hunyuan)$")
    shots: List[ShotConfig]

    @field_validator("shots")
    @classmethod
    def non_empty(cls, value: List[ShotConfig]) -> List[ShotConfig]:
        if not value:
            raise ValueError("Render plan contains no shots")
        return value


def load_plan(path: Path) -> RenderPlan:
    data = load_json(path)
    return RenderPlan.model_validate(data)
