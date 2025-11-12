#!/usr/bin/env python3
"""Render hold-out rollout JSONL traces into aggregated GIFs."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from rl.env.config import BoardShape, EnvConfig, MaskConfig, ObservationSpec, RewardPreset, DEFAULT_OBSERVATION
from rl.env.env import FlowFreeEnv

INFO_PANEL_WIDTH = 160


def build_env(puzzle: str, width: int, height: int, color_count: int, reward_params: Dict[str, float]) -> FlowFreeEnv:
    preset = RewardPreset(
        name="potential",
        components=("potential", "completion", "constraints"),
        params=reward_params,
    )
    config = EnvConfig(
        shape=BoardShape(width=width, height=height, color_count=color_count),
        puzzle=puzzle,
        reward=preset,
        observation=ObservationSpec(channels=DEFAULT_OBSERVATION.channels),
        mask=MaskConfig(),
    )
    return FlowFreeEnv(config)


def replay_episode(env: FlowFreeEnv, frames: Sequence[str]) -> List[float]:
    if not frames:
        return []
    env.reset()
    first = env.board_string
    if first != frames[0]:
        raise RuntimeError("Initial frame does not match environment state")
    cumulative = [0.0]
    for target in frames[1:]:
        if env.board_string == target:
            cumulative.append(cumulative[-1])
            continue
        mask = env.action_mask
        legal = np.where(mask > 0.0)[0]
        matched = False
        for action in legal:
            candidate = copy.deepcopy(env)
            _, reward, _, _, _ = candidate.step(int(action))
            if candidate.board_string == target:
                env = candidate
                cumulative.append(cumulative[-1] + reward)
                matched = True
                break
        if not matched:
            raise RuntimeError("Unable to match frame transition; did reward parameters change?")
    return cumulative


def add_overlay(img: Image.Image, lines: List[str]) -> Image.Image:
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    x0 = img.width - INFO_PANEL_WIDTH
    draw.rectangle([(x0, 0), (img.width, img.height)], fill=(20, 20, 20, 230))
    y = 8
    for line in lines:
        draw.text((x0 + 8, y), line, fill="white", font=font)
        y += 16
    return img


def draw_board(board: str, size: int, cell_px: int = 48) -> Image.Image:
    img = Image.new("RGB", (size * cell_px + INFO_PANEL_WIDTH, size * cell_px), color="white")
    draw = ImageDraw.Draw(img)
    for idx, char in enumerate(board):
        row, col = divmod(idx, size)
        x0, y0 = col * cell_px, row * cell_px
        x1, y1 = x0 + cell_px, y0 + cell_px
        if char.lower() == "x":
            fill = (245, 245, 245)
        else:
            color_idx = int(char)
            fill = PALETTE[color_idx % len(PALETTE)]
        draw.rectangle([x0, y0, x1, y1], fill=fill, outline=(200, 200, 200))
    return img


PALETTE = [
    (255, 255, 255),  # background
    (235, 64, 52),
    (52, 152, 219),
    (46, 204, 113),
    (241, 196, 15),
    (142, 68, 173),
    (230, 126, 34),
    (26, 188, 156),
    (127, 140, 141),
]


def render_episode_frames(frames: Sequence[str], rewards: Sequence[float], episode_id: int, solved: bool, board_size: int) -> List[Image.Image]:
    rendered: List[Image.Image] = []
    total_steps = len(frames) - 1
    for idx, board in enumerate(frames):
        img = draw_board(board, board_size)
        reward_val = rewards[idx] if idx < len(rewards) else rewards[-1]
        status = "solved" if solved else "unsolved"
        info_lines = [
            f"Episode {episode_id:03d}",
            f"Step {idx}/{total_steps if total_steps>0 else 0}",
            f"Reward {reward_val:.2f}",
            status,
        ]
        rendered.append(add_overlay(img, info_lines))
    return rendered


def load_holdout_csv(csv_path: Path) -> Dict[str, Dict[str, str]]:
    mapping: Dict[str, Dict[str, str]] = {}
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            mapping[row["board_idx"]] = row
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Render hold-out rollout traces into GIFs")
    parser.add_argument("--rollout-dir", type=Path, required=True, help="Directory containing rollout JSONL/meta pairs")
    parser.add_argument("--holdout-csv", type=Path, required=True, help="Holdout metrics CSV used for this run")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where GIFs will be saved")
    parser.add_argument("--frame-duration", type=int, default=120, help="Frame duration in ms")
    parser.add_argument("--solved-limit", type=int, default=None, help="Optional max solved episodes to render")
    parser.add_argument("--unsolved-limit", type=int, default=None, help="Optional max unsolved episodes to render")
    parser.add_argument("--move-penalty", type=float, default=-0.05)
    parser.add_argument("--distance-bonus", type=float, default=0.35)
    parser.add_argument("--complete-bonus", type=float, default=1.8)
    parser.add_argument("--complete-revert-penalty", type=float, default=2.0)
    parser.add_argument("--complete-sustain-bonus", type=float, default=0.1)
    parser.add_argument("--solve-bonus", type=float, default=35.0)
    parser.add_argument("--invalid-penalty", type=float, default=-0.3)
    parser.add_argument("--disconnect-penalty", type=float, default=-0.06)
    parser.add_argument("--degree-penalty", type=float, default=-0.03)
    parser.add_argument("--unsolved-penalty", type=float, default=-5.0)
    parser.add_argument("--loop-penalty", type=float, default=-0.15)
    parser.add_argument("--loop-window", type=int, default=6)
    parser.add_argument("--progress-bonus", type=float, default=0.02)
    parser.add_argument("--skip-mismatch", action="store_true", help="Skip episodes that cannot be replayed instead of aborting")
    args = parser.parse_args()

    reward_params = {
        "move_penalty": args.move_penalty,
        "distance_scale": args.distance_bonus,
        "complete_bonus": args.complete_bonus,
        "complete_revert_penalty": args.complete_revert_penalty,
        "complete_sustain_bonus": args.complete_sustain_bonus,
        "solve_bonus": args.solve_bonus,
        "invalid_penalty": args.invalid_penalty,
        "disconnect_penalty": args.disconnect_penalty,
        "degree_penalty": args.degree_penalty,
        "unsolved_penalty": args.unsolved_penalty,
        "loop_penalty": args.loop_penalty,
        "loop_window": args.loop_window,
        "progress_bonus": args.progress_bonus,
    }

    holdout_map = load_holdout_csv(args.holdout_csv)
    solved_frames: List[Image.Image] = []
    unsolved_frames: List[Image.Image] = []

    solved_count = 0
    unsolved_count = 0

    meta_files = sorted(args.rollout_dir.glob("*.meta.json"))
    if not meta_files:
        raise SystemExit("No rollout meta files found")

    for episode_idx, meta_path in enumerate(meta_files, start=1):
        with meta_path.open() as fh:
            meta = json.load(fh)
        board_idx = meta.get("board_idx") or meta_path.stem
        row = holdout_map.get(str(board_idx))
        if row is None:
            raise SystemExit(f"Board idx {board_idx} missing from holdout CSV")

        puzzle = row["puzzle"]
        size = int(row["board_size"])
        colors = int(row["color_count"])
        env = build_env(puzzle, size, size, colors, reward_params)

        base = meta_path
        if base.suffix == ".json":
            base = base.with_suffix("")
        if base.suffix == ".meta":
            base = base.with_suffix("")
        jsonl_path = base.with_suffix(".jsonl")
        raw_lines = jsonl_path.read_text().splitlines()
        frames: List[str] = []
        if raw_lines and raw_lines[0].lstrip().startswith("{"):
            for line in raw_lines:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    frames.append(line)
                else:
                    frames.append(entry.get("board", ""))
        else:
            frames = raw_lines
        try:
            rewards = replay_episode(env, frames)
        except RuntimeError as exc:
            if args.skip_mismatch:
                print(f"[warn] Skipping episode {episode_idx} ({meta_path.name}): {exc}")
                continue
            raise SystemExit(f"Failed to replay episode {episode_idx} ({meta_path.name}): {exc}")

        rendered = render_episode_frames(frames, rewards, episode_idx, meta.get("solved", False), size)
        if meta.get("solved", False):
            if args.solved_limit and solved_count >= args.solved_limit:
                continue
            solved_frames.extend(rendered)
            solved_count += 1
        else:
            if args.unsolved_limit and unsolved_count >= args.unsolved_limit:
                continue
            unsolved_frames.extend(rendered)
            unsolved_count += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if solved_frames:
        solved_frames[0].save(
            args.output_dir / "solved.gif",
            save_all=True,
            append_images=solved_frames[1:],
            duration=args.frame_duration,
            loop=0,
        )
    if unsolved_frames:
        unsolved_frames[0].save(
            args.output_dir / "unsolved.gif",
            save_all=True,
            append_images=unsolved_frames[1:],
            duration=args.frame_duration,
            loop=0,
        )


if __name__ == "__main__":
    main()
