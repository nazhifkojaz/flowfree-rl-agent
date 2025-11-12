from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path
from typing import Iterable

from rl.env.utils import string_to_tokens

from rl.env.constants import ACTIONS_PER_COLOR
from rl.env.config import (
    BoardShape,
    EnvConfig,
    MaskConfig,
    ObservationSpec,
    RewardPreset,
    DEFAULT_OBSERVATION,
)
from rl.env.env import FlowFreeEnv
from rl.env.trace import Trajectory, TrajectoryStep, save_trajectory


TRACE_REWARD = RewardPreset(
    name="trace_generation",
    components=("potential", "completion", "constraints"),
    params={
        "move_penalty": -0.08,
        "distance_scale": 0.35,
        "complete_bonus": 3.0,
        "solve_bonus": 20.0,
        "invalid_penalty": -1.0,
        "dead_pocket_penalty": 0.0,
        "disconnect_penalty": -0.7500712307117836,
        "degree_penalty": -0.24601623361070557,
        "unsolved_penalty": -25.0,
        "undo_penalty": -0.1,
    },
)


def parse_args() -> argparse.Namespace:
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
        "--variants",
        type=int,
        default=1,
        help="Number of colour-order variants to export per puzzle",
    )
    parser.add_argument(
        "--shuffle-colors",
        action="store_true",
        help="Randomise the order in which colours are completed for each variant",
    )
    return parser.parse_args()


def read_rows(csv_path: Path, limit: int | None) -> Iterable[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for idx, row in enumerate(reader):
            if limit is not None and idx >= limit:
                break
            yield row


def build_env_config(puzzle: str, width: int, height: int, color_count: int) -> EnvConfig:
    shape = BoardShape(width=width, height=height, color_count=color_count)
    observation = ObservationSpec(
        channels=DEFAULT_OBSERVATION.channels,
        dtype=DEFAULT_OBSERVATION.dtype,
        include_temporal_planes=DEFAULT_OBSERVATION.include_temporal_planes,
    )
    return EnvConfig(
        shape=shape,
        puzzle=puzzle,
        reward=TRACE_REWARD,
        observation=observation,
        mask=MaskConfig(),
    )


def build_env(puzzle: str, width: int, height: int, color_count: int) -> FlowFreeEnv:
    config = build_env_config(puzzle, width, height, color_count)
    return FlowFreeEnv(config)


def hash_puzzle(puzzle: str) -> str:
    return hashlib.sha1(puzzle.encode("utf-8")).hexdigest()[:12]


def neighbors(idx: int, width: int, height: int) -> list[int]:
    r, c = divmod(idx, width)
    out: list[int] = []
    if r > 0:
        out.append(idx - width)
    if r < height - 1:
        out.append(idx + width)
    if c > 0:
        out.append(idx - 1)
    if c < width - 1:
        out.append(idx + 1)
    return out


def encode_move(from_idx: int, to_idx: int, color: int, width: int) -> int:
    fr, fc = divmod(from_idx, width)
    tr, tc = divmod(to_idx, width)
    dr, dc = tr - fr, tc - fc
    if (dr, dc) == (-1, 0):
        dir_idx = 0
    elif (dr, dc) == (0, 1):
        dir_idx = 1
    elif (dr, dc) == (1, 0):
        dir_idx = 2
    elif (dr, dc) == (0, -1):
        dir_idx = 3
    else:
        raise ValueError("Move is not axis-aligned between adjacent cells")
    return (color - 1) * ACTIONS_PER_COLOR + dir_idx


def extract_endpoints(
    puzzle: list[str], width: int, height: int, color_count: int
) -> dict[int, tuple[int, int]]:
    endpoint_map: dict[int, list[int]] = {c: [] for c in range(1, color_count + 1)}
    for idx, tok in enumerate(puzzle):
        if tok.lower() == "x":
            continue
        endpoint_map[int(tok)].append(idx)

    endpoints: dict[int, tuple[int, int]] = {}
    for color, locs in endpoint_map.items():
        if len(locs) == 2:
            endpoints[color] = (locs[0], locs[1])
    return endpoints


def build_color_path(
    color: int,
    endpoints: tuple[int, int],
    final_values: list[int],
    width: int,
    height: int,
) -> list[int]:
    start_idx, goal_idx = endpoints
    color_cells = [i for i, v in enumerate(final_values) if v == color]
    if not color_cells:
        return []

    path: list[int] = []

    def dfs(current: int, prev: int | None) -> bool:
        path.append(current)
        if current == goal_idx:
            return True
        for nb in neighbors(current, width, height):
            if nb == prev:
                continue
            if final_values[nb] != color:
                continue
            if dfs(nb, current):
                return True
        path.pop()
        return False

    if not dfs(start_idx, None):
        raise ValueError(f"Unable to trace path for color {color}")
    if set(path) != set(color_cells):
        raise ValueError(f"Path for color {color} does not cover all cells")
    return path


def solution_values_from_string(
    puzzle_tokens: list[str],
    solved_tokens: list[str],
    width: int,
    height: int,
    color_count: int,
) -> list[int]:
    if len(solved_tokens) != width * height:
        raise ValueError("Solved board length mismatch")
    mapping: dict[int, int] = {}
    reverse: dict[int, int] = {}
    for idx, tok in enumerate(puzzle_tokens):
        if tok.lower() == "x":
            continue
        s_val = int(solved_tokens[idx])
        p_val = int(tok)
        if s_val in mapping and mapping[s_val] != p_val:
            raise ValueError("Solved board does not respect puzzle endpoints")
        if p_val in reverse and reverse[p_val] != s_val:
            raise ValueError("Solved board does not respect puzzle endpoints")
        mapping[s_val] = p_val
        reverse[p_val] = s_val

    remapped: list[int] = []
    for tok in solved_tokens:
        val = int(tok)
        remapped.append(mapping.get(val, val))
    return remapped


def main() -> None:
    args = parse_args()
    rows = list(read_rows(args.csv, args.limit))
    if not rows:
        print("No puzzles to process.")
        return

    summary = {"success": 0, "failure": 0, "skipped": 0}
    failures: list[tuple[str, str]] = []

    for row in rows:
        size = int(row["BoardSize"])
        if size > args.max_size:
            continue
        color_count = int(row["ColorCount"])
        if args.max_colors is not None and color_count > args.max_colors:
            continue

        puzzle = row["InitialPuzzle"]
        complete = row["CompletePuzzle"]

        initial_tokens = string_to_tokens(puzzle, size, size, color_count)
        solved_tokens = string_to_tokens(complete, size, size, color_count)
        if initial_tokens is None or solved_tokens is None:
            summary["failure"] += 1
            failures.append((row.get("board_idx", "?"), "failed to parse puzzle/solution"))
            continue

        total_cells = size * size
        if len(initial_tokens) != total_cells or len(solved_tokens) != total_cells:
            summary["failure"] += 1
            failures.append((row.get("board_idx", "?"), "token length mismatch"))
            continue

        mismatch = False
        for idx, tok in enumerate(initial_tokens):
            if tok.lower() != "x" and tok != solved_tokens[idx]:
                mismatch = True
                break
        if mismatch:
            summary["failure"] += 1
            failures.append((row.get("board_idx", "?"), "solution does not respect endpoints"))
            continue

        try:
            final_values = solution_values_from_string(
                initial_tokens, solved_tokens, size, size, color_count
            )
        except Exception as exc:  # noqa: BLE001
            summary["failure"] += 1
            failures.append((row.get("board_idx", "?"), str(exc)))
            continue

        endpoints = extract_endpoints(initial_tokens, size, size, color_count)
        try:
            paths: dict[int, list[int]] = {}
            for color in range(1, color_count + 1):
                if color not in endpoints:
                    continue
                path = build_color_path(color, endpoints[color], final_values, size, size)
                paths[color] = path
        except Exception as exc:  # noqa: BLE001
            summary["failure"] += 1
            failures.append((row.get("board_idx", "?"), str(exc)))
            continue

        color_keys = sorted(paths.keys())
        final_string = "".join(solved_tokens)
        variant_orders: list[list[int]] = []
        base_order = color_keys.copy()
        signature_seed = int(hash_puzzle(puzzle), 16)
        total_variants = max(1, args.variants)
        if args.shuffle_colors:
            seen_orders: set[tuple[int, ...]] = set()
            for variant_idx in range(total_variants):
                rng = random.Random(signature_seed + variant_idx)
                shuffled = base_order.copy()
                rng.shuffle(shuffled)
                tup = tuple(shuffled)
                if tup in seen_orders:
                    continue
                seen_orders.add(tup)
                variant_orders.append(shuffled)
            if not variant_orders:
                variant_orders.append(base_order)
        else:
            variant_orders.append(base_order)

        for variant_idx, order in enumerate(variant_orders):
            env = build_env(puzzle, size, size, color_count)
            obs, _ = env.reset()
            steps: list[TrajectoryStep] = []
            terminated = False
            failure = False

            for color in order:
                path = paths.get(color)
                if not path or len(path) < 2:
                    continue
                current = path[0]
                for next_idx in path[1:]:
                    if (
                        initial_tokens[next_idx].lower() != "x"
                        and next_idx not in endpoints.get(color, ())
                    ):
                        current = next_idx
                        continue
                    action = encode_move(current, next_idx, color, size)
                    mask = obs["action_mask"]
                    if action >= len(mask) or mask[action] <= 0.0:
                        failure = True
                        break
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    steps.append(
                        TrajectoryStep(
                            action=action,
                            reward=reward,
                            terminated=terminated,
                            truncated=truncated,
                            info=info,
                            observation=next_obs,
                        )
                    )
                    obs = next_obs
                    current = next_idx
                    if terminated:
                        break
                if failure or terminated:
                    break
            if not failure and env.board_string == final_string:
                terminated = True

            if not terminated:
                summary["failure"] += 1
                failures.append(
                    (row.get("board_idx", "?"), f"could not reconstruct sequence (variant {variant_idx})")
                )
                continue

            trajectory = Trajectory(config=build_env_config(puzzle, size, size, color_count), steps=steps)

            signature = hash_puzzle(puzzle)
            suffix = f"_v{variant_idx:02d}" if (args.shuffle_colors or total_variants > 1) else ""
            out_path = (
                args.out_dir
                / args.solver_name
                / f"{args.solver_name}_{size}x{size}_{color_count}c_{signature}{suffix}.json"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if out_path.exists() and not args.force:
                summary["skipped"] += 1
                if args.verbose:
                    print(f"[skipped] #{row.get('board_idx', '?')} -> {out_path}")
                continue

            save_trajectory(trajectory, out_path)
            summary["success"] += 1
            if args.verbose:
                order_desc = ",".join(map(str, order))
                print(f"[ok] #{row.get('board_idx', '?')} -> {out_path} (order {order_desc})")

    print(
        "Processed {total} puzzles (success={success}, skipped={skipped}, failure={failure}).".format(
            total=len(rows), **summary
        )
    )
    if failures:
        print("Failures:")
        for board_idx, message in failures[:20]:
            print(f"  #{board_idx}: {message}")
        if len(failures) > 20:
            print(f"  … {len(failures) - 20} more failures")


if __name__ == "__main__":
    main()
