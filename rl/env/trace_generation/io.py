"""I/O utilities for reading puzzles and saving trajectories."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable


def read_rows(csv_path: Path, limit: int | None) -> Iterable[dict[str, str]]:
    """Read rows from a CSV file containing puzzles.

    Args:
        csv_path: Path to CSV file
        limit: Optional maximum number of rows to read

    Yields:
        Dictionary for each row with CSV column names as keys
    """
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            yield row


__all__ = ["read_rows"]
