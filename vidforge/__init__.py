"""VidForge: modular long-form video generation toolkit."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("vidforge")
except PackageNotFoundError:  # pragma: no cover - fallback for editable installs
    __version__ = "0.1.0"

__all__ = ["__version__"]
