"""Command-line interface for trace generation."""

from __future__ import annotations

import argparse
from pathlib import Path

from rl.env.trace_generation.config import TraceGenConfig
from rl.env.trace_generation.io import read_rows
from rl.env.trace_generation.processor import TraceProcessor
from rl.env.trace_generation.strategies import get_strategy


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        argv: Optional argument list (defaults to sys.argv)

    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        description="Generate trajectories by replaying completed boards colour-by-colour."
    )
    parser.add_argument("--csv", type=Path, required=True, help="CSV containing puzzles")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/"),
        help="Directory to store generated trajectories",
    )
    parser.add_argument(
        "--solver-name",
        type=str,
        default="rl_traces",
        help="Prefix used for saved trajectory files",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of rows processed",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=8,
        help="Maximum board size to process",
    )
    parser.add_argument(
        "--max-colors",
        type=int,
        default=None,
        help="Optional cap on colour count",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing trajectory files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log per-puzzle progress",
    )
    parser.add_argument(
        "--completion-mode",
        dest="completion_modes",
        action="append",
        choices=("normal", "longest", "onedistance", "blocked", "oneattime"),
        help=(
            "Completion strategy to run. Can be provided multiple times to generate"
            " traces for several modes in one invocation. Defaults to 'normal' if"
            " omitted."
        ),
    )
    parser.add_argument(
        "--variants",
        type=int,
        default=1,
        help="(normal mode only) number of colour-order variants to export per puzzle",
    )
    parser.add_argument(
        "--shuffle-colors",
        action="store_true",
        help="(normal mode only) randomise the order in which colours are completed for each variant",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Main entry point for trace generation CLI.

    Args:
        argv: Optional argument list (defaults to sys.argv)
    """
    args = parse_args(argv)
    completion_modes = args.completion_modes or ["normal"]

    rows = list(read_rows(args.csv, args.limit))
    if not rows:
        print("No puzzles to process.")
        return

    for mode in completion_modes:
        print(f"\n=== Completion mode: {mode} ===")
        config = TraceGenConfig(
            out_dir=args.out_dir,
            solver_name=args.solver_name,
            max_size=args.max_size,
            max_colors=args.max_colors,
            force_overwrite=args.force,
            verbose=args.verbose,
            completion_mode=mode,
            variants=args.variants,
            shuffle_colors=args.shuffle_colors,
        )

        strategy = get_strategy(mode)
        if (args.variants > 1 or args.shuffle_colors) and not strategy.supports_variants():
            print(
                f"[warn] --variants/--shuffle-colors are only applied in 'normal' completion mode. "
                f"Ignoring for '{mode}'."
            )

        processor = TraceProcessor(config, strategy)
        for row in rows:
            processor.process_puzzle(row)
        processor.print_summary()


__all__ = ["main", "parse_args"]


if __name__ == "__main__":
    main()
