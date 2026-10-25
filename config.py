"""Local paths. Relative paths are resolved from the project root."""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


def resolve_path(value):
    """Resolve a path relative to the project root when needed.

    Args:
        value (str or Path): Absolute or project-relative path; ~ expands to the home folder.
    """
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


SC2_PATH = resolve_path(os.getenv("SC2PATH", "C:/Program Files (x86)/StarCraft II"))
REPLAY_DIR = resolve_path(os.getenv("REPLAY_DIR", "data_sample"))
PROCESSED_DIR = resolve_path(os.getenv("PROCESSED_DIR", "data/processed"))
