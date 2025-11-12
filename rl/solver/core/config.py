from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SeedConfig:
    """Shared seeding options for trainers."""

    seed: int = 7
    torch_deterministic: bool = False


@dataclass(slots=True)
class RunPaths:
    """Common filesystem layout for training runs."""

    log_root: Path = Path("flowfree/logs/rl_training")
    output: Path = Path("models")
    tensorboard_dir: Path | None = None


__all__ = ["SeedConfig", "RunPaths"]
