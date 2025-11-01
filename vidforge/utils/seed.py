"""Seed helpers for reproducible generation."""
from __future__ import annotations

import hashlib
from typing import Optional


def prompt_to_seed(prompt: str, salt: Optional[str] = None) -> int:
    payload = prompt if salt is None else f"{salt}|{prompt}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def resolve_seed(explicit: Optional[int], prompt: str, salt: Optional[str] = None) -> int:
    return explicit if explicit is not None else prompt_to_seed(prompt, salt=salt)
