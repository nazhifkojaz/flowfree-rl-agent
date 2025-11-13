"""TraceProcessor for handling puzzle processing and trajectory generation."""

from __future__ import annotations

import random
from pathlib import Path

from rl.env.config import BoardShape, EnvConfig, MaskConfig, ObservationSpec
from rl.env.env import FlowFreeEnv
from rl.env.trace import Trajectory, TrajectoryStep, save_trajectory
from rl.env.utils import string_to_tokens
from rl.env.trace_generation.config import TraceGenConfig
from rl.env.trace_generation.grid_utils import encode_move, hash_puzzle
from rl.env.trace_generation.path_builder import (
    build_color_path,
    extract_endpoints,
    solution_values_from_string,
)
from rl.env.trace_generation.strategies import CompletionContext, CompletionStrategy


class TraceProcessor:
    """Handles puzzle processing and trajectory generation.

    This class orchestrates the entire pipeline from loading a puzzle row to
    generating and saving trajectory files.
    """

    def __init__(self, config: TraceGenConfig, strategy: CompletionStrategy):
        """Initialize the processor.

        Args:
            config: Configuration for trace generation
            strategy: Completion strategy to use
        """
        self.config = config
        self.strategy = strategy
        self.stats = {"success": 0, "failure": 0, "skipped": 0}
        self.failures: list[tuple[str, str]] = []

    def process_puzzle(self, row: dict[str, str]) -> list[Path]:
        """Process a single puzzle and generate trajectory files.

        Args:
            row: CSV row dict with puzzle data

        Returns:
            List of paths to saved trajectory files (may be empty on failure/skip)
        """
        # 1. Validate size and colors
        size = int(row["BoardSize"])
        if size > self.config.max_size:
            return []

        color_count = int(row["ColorCount"])
        if self.config.max_colors is not None and color_count > self.config.max_colors:
            return []

        puzzle = row["InitialPuzzle"]
        complete = row["CompletePuzzle"]
        board_idx = row.get("board_idx", "?")

        # 2. Parse and validate solution
        result = self.validate_solution(puzzle, complete, size, size, color_count)
        if result is None:
            self.stats["failure"] += 1
            self.failures.append((board_idx, "solution validation failed"))
            return []

        final_values, paths = result
        endpoints = extract_endpoints(
            string_to_tokens(puzzle, size, size, color_count) or [], size, size, color_count
        )

        # 3. Build completion order and variants
        ctx = CompletionContext(
            paths=paths,
            endpoints=endpoints,
            width=size,
            height=size,
            color_count=color_count,
        )
        base_order = self.strategy.build_color_order(ctx)
        variant_orders = self._build_variant_orders(base_order, puzzle)

        # 4. Generate trajectories for each variant
        saved_paths: list[Path] = []
        for variant_idx, order in enumerate(variant_orders):
            traj_path = self._generate_trajectory(
                puzzle, complete, size, color_count, order, paths, endpoints, variant_idx, board_idx
            )
            if traj_path:
                saved_paths.append(traj_path)

        return saved_paths

    def validate_solution(
        self, puzzle: str, solution: str, width: int, height: int, color_count: int
    ) -> tuple[list[int], dict[int, list[int]]] | None:
        """Validate solution and extract paths.

        Args:
            puzzle: Initial puzzle string
            solution: Completed puzzle string
            width: Grid width
            height: Grid height
            color_count: Number of colors

        Returns:
            Tuple of (final_values, paths) or None on failure
        """
        # Parse tokens
        initial_tokens = string_to_tokens(puzzle, width, height, color_count)
        solved_tokens = string_to_tokens(solution, width, height, color_count)
        if initial_tokens is None or solved_tokens is None:
            return None

        total_cells = width * height
        if len(initial_tokens) != total_cells or len(solved_tokens) != total_cells:
            return None

        # Check endpoints are respected
        for idx, tok in enumerate(initial_tokens):
            if tok.lower() != "x" and tok != solved_tokens[idx]:
                return None

        # Remap solution to puzzle colors
        try:
            final_values = solution_values_from_string(
                initial_tokens, solved_tokens, width, height, color_count
            )
        except Exception:  # noqa: BLE001
            return None

        # Reconstruct paths
        endpoints = extract_endpoints(initial_tokens, width, height, color_count)
        try:
            paths: dict[int, list[int]] = {}
            for color in range(1, color_count + 1):
                if color not in endpoints:
                    continue
                path = build_color_path(color, endpoints[color], final_values, width, height)
                paths[color] = path
        except Exception:  # noqa: BLE001
            return None

        return final_values, paths

    def _generate_trajectory(
        self,
        puzzle: str,
        solution: str,
        size: int,
        color_count: int,
        order: list[int],
        paths: dict[int, list[int]],
        endpoints: dict[int, tuple[int, int]],
        variant_idx: int,
        board_idx: str,
    ) -> Path | None:
        """Generate and save a single trajectory variant.

        Returns:
            Path to saved file or None on failure
        """
        # Build environment
        env = self._build_env(puzzle, size, size, color_count)
        obs, _ = env.reset()

        # Build edge schedule
        ctx = CompletionContext(
            paths=paths,
            endpoints=endpoints,
            width=size,
            height=size,
            color_count=color_count,
        )
        schedule = self.strategy.build_edge_schedule(order, ctx)

        # Replay trajectory
        result = self._replay_trajectory(env, schedule, paths, endpoints, obs)
        if result is None:
            self.stats["failure"] += 1
            self.failures.append((board_idx, f"replay failed (variant {variant_idx})"))
            return None

        steps, terminated = result

        # Verify completion
        solved_tokens = string_to_tokens(solution, size, size, color_count)
        final_string = "".join(solved_tokens) if solved_tokens else ""
        if not terminated or env.board_string != final_string:
            self.stats["failure"] += 1
            self.failures.append((board_idx, f"incomplete replay (variant {variant_idx})"))
            return None

        # Build and save trajectory
        env_config = self._build_env_config(puzzle, size, size, color_count)
        trajectory = Trajectory(config=env_config, steps=steps)

        signature = hash_puzzle(puzzle)
        total_variants = len(self._build_variant_orders(self.strategy.build_color_order(ctx), puzzle))
        suffix = f"_v{variant_idx:02d}" if (self.config.shuffle_colors or total_variants > 1) else ""
        mode_tag = self._mode_suffix()
        out_path = (
            self.config.out_dir
            / self.config.solver_name
            / f"{self.config.solver_name}_{size}x{size}_{color_count}c_{signature}{mode_tag}{suffix}.json"
        )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists() and not self.config.force_overwrite:
            self.stats["skipped"] += 1
            if self.config.verbose:
                print(f"[skipped] #{board_idx} -> {out_path}")
            return None

        save_trajectory(trajectory, out_path)
        self.stats["success"] += 1
        if self.config.verbose:
            order_desc = ",".join(map(str, order))
            print(f"[ok] #{board_idx} -> {out_path} (order {order_desc})")

        return out_path

    def _mode_suffix(self) -> str:
        """Return a filename-safe suffix encoding the completion mode."""
        mode = (self.config.completion_mode or "").lower()
        if not mode:
            return ""
        safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in mode)
        return f"__{safe}" if safe else ""

    def _replay_trajectory(
        self,
        env: FlowFreeEnv,
        schedule: list[tuple[int, int]],
        paths: dict[int, list[int]],
        endpoints: dict[int, tuple[int, int]],
        obs: dict,
    ) -> tuple[list[TrajectoryStep], bool] | None:
        """Replay a schedule and collect trajectory steps.

        Returns:
            Tuple of (steps, terminated) or None on failure
        """
        steps: list[TrajectoryStep] = []
        terminated = False
        initial_tokens = string_to_tokens(
            env.config.puzzle, env.config.shape.width, env.config.shape.height, env.config.shape.color_count
        ) or []

        for color, edge_idx in schedule:
            path = paths.get(color)
            if not path or edge_idx >= len(path) - 1:
                continue

            current = path[edge_idx]
            next_idx = path[edge_idx + 1]

            # Skip if next cell is an occupied endpoint
            if initial_tokens[next_idx].lower() != "x" and next_idx not in endpoints.get(color, ()):
                continue

            action = encode_move(current, next_idx, color, env.config.shape.width)
            mask = obs["action_mask"]

            if action >= len(mask) or mask[action] <= 0.0:
                return None  # Invalid action

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

            if terminated:
                break

        return steps, terminated

    def _build_variant_orders(self, base_order: list[int], puzzle: str) -> list[list[int]]:
        """Build variant color orders (for strategies that support it)."""
        if not self.strategy.supports_variants():
            return [base_order]

        if not self.config.shuffle_colors:
            return [base_order] * self.config.variants

        # Generate shuffled variants
        signature_seed = int(hash_puzzle(puzzle), 16)
        variant_orders: list[list[int]] = []
        seen_orders: set[tuple[int, ...]] = set()

        for variant_idx in range(max(1, self.config.variants)):
            rng = random.Random(signature_seed + variant_idx)
            shuffled = base_order.copy()
            rng.shuffle(shuffled)
            tup = tuple(shuffled)

            if tup in seen_orders:
                continue
            seen_orders.add(tup)
            variant_orders.append(shuffled)

        return variant_orders if variant_orders else [base_order]

    def _build_env_config(self, puzzle: str, width: int, height: int, color_count: int) -> EnvConfig:
        """Build environment configuration."""
        from rl.env.config import DEFAULT_OBSERVATION

        shape = BoardShape(width=width, height=height, color_count=color_count)
        observation = ObservationSpec(
            channels=DEFAULT_OBSERVATION.channels,
            dtype=DEFAULT_OBSERVATION.dtype,
            include_temporal_planes=DEFAULT_OBSERVATION.include_temporal_planes,
        )
        return EnvConfig(
            shape=shape,
            puzzle=puzzle,
            reward=self.config.reward_preset,
            observation=observation,
            mask=MaskConfig(),
        )

    def _build_env(self, puzzle: str, width: int, height: int, color_count: int) -> FlowFreeEnv:
        """Build FlowFreeEnv instance."""
        config = self._build_env_config(puzzle, width, height, color_count)
        return FlowFreeEnv(config)

    def print_summary(self) -> None:
        """Print final processing statistics."""
        total = sum(self.stats.values())
        print(
            f"Processed {total} puzzles "
            f"(success={self.stats['success']}, "
            f"skipped={self.stats['skipped']}, "
            f"failure={self.stats['failure']})."
        )
        if self.failures:
            print("Failures:")
            for board_idx, message in self.failures[:20]:
                print(f"  #{board_idx}: {message}")
            if len(self.failures) > 20:
                print(f"  … {len(self.failures) - 20} more failures")


__all__ = ["TraceProcessor"]
