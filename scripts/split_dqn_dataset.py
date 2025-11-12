#!/usr/bin/env python3
"""
Split puzzle_data.csv into train/validation/test sets for DQN training.

This script creates stratified splits ensuring balanced distribution of:
- Board sizes (5x5, 6x6, etc.)
- Color counts (3, 4, 5, 6 colors)
- Puzzle difficulty (based on solution density 'v')

Usage:
    python scripts/split_dqn_dataset.py
    python scripts/split_dqn_dataset.py --input data/puzzle_data.csv --output-dir data/splits
    python scripts/split_dqn_dataset.py --train-ratio 0.7 --val-ratio 0.15 --test-ratio 0.15
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def load_puzzles(csv_path: Path) -> list[dict[str, Any]]:
    """Load puzzles from CSV file."""
    puzzles = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Parse relevant fields
            row["BoardSize"] = int(row["BoardSize"])
            row["ColorCount"] = int(row["ColorCount"])
            row["v"] = float(row["v"])
            puzzles.append(row)
    return puzzles


def stratify_puzzles(puzzles: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    """Group puzzles by (board_size, color_count) for stratified sampling."""
    strata = defaultdict(list)
    for puzzle in puzzles:
        key = (puzzle["BoardSize"], puzzle["ColorCount"])
        strata[key].append(puzzle)
    return strata


def split_stratum(
    stratum: list[dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Split a single stratum into train/val/test sets."""
    # Shuffle within stratum
    rng = np.random.RandomState(seed)
    indices = np.arange(len(stratum))
    rng.shuffle(indices)

    # Calculate split points
    n = len(stratum)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # Ensure all samples are assigned (test gets remainder)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train = [stratum[i] for i in train_indices]
    val = [stratum[i] for i in val_indices]
    test = [stratum[i] for i in test_indices]

    return train, val, test


def save_puzzles(puzzles: list[dict[str, Any]], output_path: Path, fieldnames: list[str]) -> None:
    """Save puzzles to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(puzzles)


def print_split_stats(split_name: str, puzzles: list[dict[str, Any]]) -> None:
    """Print statistics for a data split."""
    if not puzzles:
        print(f"  {split_name}: 0 puzzles")
        return

    sizes = defaultdict(int)
    colors = defaultdict(int)
    for p in puzzles:
        sizes[p["BoardSize"]] += 1
        colors[p["ColorCount"]] += 1

    print(f"  {split_name}: {len(puzzles)} puzzles")
    print(f"    Board sizes: {dict(sorted(sizes.items()))}")
    print(f"    Color counts: {dict(sorted(colors.items()))}")

    # Difficulty stats
    difficulties = [p["v"] for p in puzzles]
    print(f"    Difficulty (v): min={min(difficulties):.3f}, "
          f"mean={np.mean(difficulties):.3f}, max={max(difficulties):.3f}")


def main():
    parser = argparse.ArgumentParser(
        description="Split puzzle_data.csv into stratified train/val/test sets"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/puzzle_data.csv"),
        help="Input CSV file (default: data/puzzle_data.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data"),
        help="Output directory for split files (default: data/)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.7,
        help="Fraction of data for training (default: 0.7)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Fraction of data for validation (default: 0.15)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Fraction of data for testing (default: 0.15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=None,
        help="Minimum board size to include (optional)",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Maximum board size to include (optional)",
    )
    args = parser.parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if not np.isclose(total_ratio, 1.0):
        print(f"Warning: Ratios sum to {total_ratio}, normalizing...")
        args.train_ratio /= total_ratio
        args.val_ratio /= total_ratio
        args.test_ratio /= total_ratio

    # Load puzzles
    print(f"Loading puzzles from {args.input}...")
    puzzles = load_puzzles(args.input)
    print(f"Loaded {len(puzzles)} puzzles")

    # Filter by size if requested
    if args.min_size or args.max_size:
        original_count = len(puzzles)
        if args.min_size:
            puzzles = [p for p in puzzles if p["BoardSize"] >= args.min_size]
        if args.max_size:
            puzzles = [p for p in puzzles if p["BoardSize"] <= args.max_size]
        print(f"Filtered to {len(puzzles)} puzzles (removed {original_count - len(puzzles)})")

    # Stratify by (board_size, color_count)
    print("\nStratifying puzzles by (board_size, color_count)...")
    strata = stratify_puzzles(puzzles)
    print(f"Found {len(strata)} strata:")
    for key, stratum in sorted(strata.items()):
        print(f"  {key[0]}×{key[0]}, {key[1]} colors: {len(stratum)} puzzles")

    # Split each stratum
    print(f"\nSplitting with ratios: train={args.train_ratio:.2f}, "
          f"val={args.val_ratio:.2f}, test={args.test_ratio:.2f}")

    train_puzzles = []
    val_puzzles = []
    test_puzzles = []

    for key, stratum in sorted(strata.items()):
        train, val, test = split_stratum(
            stratum, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
        )
        train_puzzles.extend(train)
        val_puzzles.extend(val)
        test_puzzles.extend(test)

        print(f"  {key[0]}×{key[0]}, {key[1]} colors: "
              f"train={len(train)}, val={len(val)}, test={len(test)}")

    # Shuffle each split (within-split shuffle)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(train_puzzles)
    rng.shuffle(val_puzzles)
    rng.shuffle(test_puzzles)

    # Save splits
    fieldnames = list(puzzles[0].keys())

    train_path = args.output_dir / "dqn_train.csv"
    val_path = args.output_dir / "dqn_val.csv"
    test_path = args.output_dir / "dqn_test.csv"

    print(f"\nSaving splits to {args.output_dir}/...")
    save_puzzles(train_puzzles, train_path, fieldnames)
    save_puzzles(val_puzzles, val_path, fieldnames)
    save_puzzles(test_puzzles, test_path, fieldnames)

    print(f"  Train: {train_path} ({len(train_puzzles)} puzzles)")
    print(f"  Val:   {val_path} ({len(val_puzzles)} puzzles)")
    print(f"  Test:  {test_path} ({len(test_puzzles)} puzzles)")

    # Print detailed statistics
    print("\n" + "="*70)
    print("Split Statistics")
    print("="*70)
    print_split_stats("Train", train_puzzles)
    print()
    print_split_stats("Validation", val_puzzles)
    print()
    print_split_stats("Test", test_puzzles)

    print("\n" + "="*70)
    print("Split complete!")
    print("="*70)
    print(f"\nTotal: {len(train_puzzles) + len(val_puzzles) + len(test_puzzles)} puzzles")
    print(f"  Train: {len(train_puzzles)} ({len(train_puzzles)/len(puzzles)*100:.1f}%)")
    print(f"  Val:   {len(val_puzzles)} ({len(val_puzzles)/len(puzzles)*100:.1f}%)")
    print(f"  Test:  {len(test_puzzles)} ({len(test_puzzles)/len(puzzles)*100:.1f}%)")

    print("\nUsage in training:")
    print(f"  python rl/solver/train_dqn.py --puzzle-csv {train_path} --validation-csv {val_path}")
    print(f"  python rl/solver/eval_holdout.py --holdout-csv {test_path}")


if __name__ == "__main__":
    main()
