"""Filesystem utilities."""
from __future__ import annotations

import contextlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterator

from rich.console import Console

console = Console()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@contextlib.contextmanager
def atomic_write(path: Path, mode: str = "w", **kwargs: Any) -> Iterator[Any]:
    ensure_dir(path.parent)
    with tempfile.NamedTemporaryFile(mode=mode, delete=False, dir=path.parent, **kwargs) as tmp:
        tmp_path = Path(tmp.name)
        try:
            yield tmp
            tmp.flush()
            os.fsync(tmp.fileno())
            tmp.close()
            tmp_path.replace(path)
        except Exception:  # pragma: no cover - rethrow after cleanup
            tmp.close()
            tmp_path.unlink(missing_ok=True)
            raise


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    with atomic_write(path, mode="w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.write("\n")


def resolve_cache_subdir(subdir: str) -> Path:
    from vidforge.config import settings

    target = settings.cache_dir / subdir
    ensure_dir(target)
    return target


def log_path(path: Path) -> None:
    console.log(f"[bold green]✔[/] {path}")
