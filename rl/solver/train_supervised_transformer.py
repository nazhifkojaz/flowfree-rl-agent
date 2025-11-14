"""Supervised imitation training that mirrors the FlowQNetwork architecture.

This script trains the full FlowQNetwork (backbone + Transformer action head)
so the resulting checkpoint can be loaded directly before DQN fine-tuning.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from rl.solver.constants import MAX_CHANNELS
from rl.solver.dataset import SupervisedTrajectoryDataset, discover_trace_files
from rl.solver.policies.q_network import FlowQNetwork
from rl.solver.observation import encode_observation, mask_to_tensor
from rl.env.env import FlowFreeEnv
from rl.env.config import EnvConfig, BoardShape, POTENTIAL_REWARD, DEFAULT_OBSERVATION


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FlowQNetwork via supervised imitation.")
    parser.add_argument(
        "--traces-dir",
        type=Path,
        nargs="+",
        default=[Path("data/rl_traces")],
        help="Directories containing trajectory JSON traces.",
    )
    parser.add_argument("--output", type=Path, default=Path("models/dqn_supervised_transformer.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-traces", type=int, default=None, help="Optional cap on the number of trace files.")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of data reserved for validation.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-dueling", action="store_true", help="Match dueling head used in DQN.")
    parser.add_argument("--eval-puzzles", type=Path, default=None, help="CSV file with puzzles for solve rate evaluation")
    parser.add_argument("--eval-interval", type=int, default=5, help="Evaluate solve rate every N epochs")
    parser.add_argument("--eval-episodes", type=int, default=20, help="Number of puzzles to evaluate per interval")
    parser.add_argument("--max-steps", type=int, default=50, help="Max steps per evaluation episode")
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(batch):
    states = torch.stack([ex.state for ex in batch])
    head_masks = torch.stack([ex.head_mask for ex in batch])
    target_masks = torch.stack([ex.target_mask for ex in batch])
    color_counts = torch.tensor([int(ex.colour_count.item()) for ex in batch], dtype=torch.long)
    action_masks = torch.stack([ex.action_mask for ex in batch]).to(torch.bool)
    actions = torch.tensor([ex.action for ex in batch], dtype=torch.long)
    return states, head_masks, target_masks, color_counts, action_masks, actions


def evaluate_solve_rate(
    model: FlowQNetwork,
    puzzles_csv: Path,
    device: torch.device,
    max_episodes: int = 20,
    max_steps: int = 50,
    seed: int = 42,
) -> dict[str, float]:
    """Evaluate policy by actually solving puzzles."""
    if not puzzles_csv.exists():
        return {"solve_rate": 0.0, "avg_reward": 0.0, "puzzles_tested": 0}

    df = pd.read_csv(puzzles_csv)
    if len(df) == 0:
        return {"solve_rate": 0.0, "avg_reward": 0.0, "puzzles_tested": 0}

    # Sample puzzles
    np.random.seed(seed)
    sample_size = min(max_episodes, len(df))
    sampled = df.sample(n=sample_size, random_state=seed)

    model.eval()
    solves = 0
    total_reward = 0.0

    with torch.no_grad():
        for _, row in sampled.iterrows():
            # Create environment for this puzzle
            # Handle both BoardSize (square) and Width/Height (rectangular) formats
            if "BoardSize" in row:
                size = int(row["BoardSize"])
                board_shape = BoardShape(width=size, height=size, color_count=int(row["ColorCount"]))
            else:
                board_shape = BoardShape(
                    width=int(row["Width"]),
                    height=int(row["Height"]),
                    color_count=int(row["ColorCount"])
                )

            env_config = EnvConfig(
                shape=board_shape,
                puzzle=row["InitialPuzzle"],
                reward=POTENTIAL_REWARD,
                observation=DEFAULT_OBSERVATION,
                max_steps=max_steps,
            )
            env = FlowFreeEnv(env_config)

            obs, info = env.reset()

            episode_reward = 0.0
            terminated = False

            for _ in range(max_steps):
                if terminated:
                    break

                # Encode observation using solver's encoding
                encoded = encode_observation(obs, device=device)
                state_tensor = encoded.state.unsqueeze(0)
                head_mask = encoded.head_mask.unsqueeze(0)
                target_mask = encoded.target_mask.unsqueeze(0)
                color_count = torch.tensor([encoded.color_count], dtype=torch.long, device=device)

                # Get action mask - use mask_to_tensor to properly pad it
                action_mask = mask_to_tensor(
                    obs["action_mask"],
                    device=device,
                    color_count=obs["color_count"]
                ).unsqueeze(0).bool()

                logits = model(
                    state_tensor,
                    head_masks=head_mask,
                    target_masks=target_mask,
                    color_counts=color_count,
                )
                logits = logits.masked_fill(~action_mask, -1e9)
                action = torch.argmax(logits, dim=-1).item()

                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward

                if terminated or truncated:
                    break

            if info.get("solved", False):
                solves += 1
            total_reward += episode_reward

    solve_rate = (solves / sample_size) * 100.0
    avg_reward = total_reward / sample_size

    return {
        "solve_rate": solve_rate,
        "avg_reward": avg_reward,
        "puzzles_tested": sample_size,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    seed_everything(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trace_files: list[Path] = []
    for root in args.traces_dir:
        trace_files.extend(discover_trace_files(root))
    if not trace_files:
        raise SystemExit("No trajectory files found. Generate traces before training.")

    trace_files = sorted(trace_files)
    if args.max_traces is not None:
        trace_files = trace_files[: args.max_traces]

    dataset = SupervisedTrajectoryDataset(trace_files)
    if len(dataset) == 0:
        raise SystemExit("Loaded dataset is empty; verify trace contents.")

    val_size = int(len(dataset) * max(0.0, min(1.0, args.val_ratio)))
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_batch,
        pin_memory=True,
    )
    val_loader = (
        DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_batch,
            pin_memory=True,
        )
        if val_dataset is not None
        else None
    )

    model = FlowQNetwork(in_channels=MAX_CHANNELS, use_dueling=args.use_dueling).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Loaded {len(trace_files)} traces → {len(dataset)} examples")
    if val_loader is not None:
        print(f"Train examples: {train_size} | Validation examples: {val_size}")

    best_val_loss = float('inf')
    best_model_path = args.output.parent / f"{args.output.stem}_best.pt"

    for epoch in range(1, args.epochs + 1):
        # Training
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        batches = 0

        for states, head_masks, target_masks, color_counts, action_masks, actions in train_loader:
            states = states.to(device)
            head_masks = head_masks.to(device)
            target_masks = target_masks.to(device)
            color_counts = color_counts.to(device)
            action_masks = action_masks.to(device)
            actions = actions.to(device)

            logits = model(
                states,
                head_masks=head_masks,
                target_masks=target_masks,
                color_counts=color_counts,
            )
            logits = logits.masked_fill(~action_masks, -1e9)
            loss = F.cross_entropy(logits, actions)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())

            # Track accuracy
            predictions = torch.argmax(logits, dim=-1)
            correct += (predictions == actions).sum().item()
            total += actions.size(0)
            batches += 1

        train_loss = epoch_loss / max(1, batches)
        train_acc = (correct / max(1, total)) * 100.0

        # Validation
        val_loss = None
        val_acc = None
        if val_loader is not None:
            model.eval()
            total_val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for states, head_masks, target_masks, color_counts, action_masks, actions in val_loader:
                    states = states.to(device)
                    head_masks = head_masks.to(device)
                    target_masks = target_masks.to(device)
                    color_counts = color_counts.to(device)
                    action_masks = action_masks.to(device)
                    actions = actions.to(device)

                    logits = model(
                        states,
                        head_masks=head_masks,
                        target_masks=target_masks,
                        color_counts=color_counts,
                    )
                    logits = logits.masked_fill(~action_masks, -1e9)
                    loss = F.cross_entropy(logits, actions)
                    total_val_loss += float(loss.item())

                    # Track accuracy
                    predictions = torch.argmax(logits, dim=-1)
                    val_correct += (predictions == actions).sum().item()
                    val_total += actions.size(0)

            val_loss = total_val_loss / max(1, len(val_loader))
            val_acc = (val_correct / max(1, val_total)) * 100.0

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.cpu().state_dict(), best_model_path)
                model.to(device)
                print(f"  → New best model saved (val_loss={val_loss:.6f})")

        # Print metrics
        if val_loss is not None:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} train_acc={train_acc:.2f}% | val_loss={val_loss:.6f} val_acc={val_acc:.2f}%")
        else:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} train_acc={train_acc:.2f}%")

        # Puzzle solve evaluation
        if args.eval_puzzles and epoch % args.eval_interval == 0:
            print(f"  Running puzzle solve evaluation...")
            eval_metrics = evaluate_solve_rate(
                model,
                args.eval_puzzles,
                device,
                max_episodes=args.eval_episodes,
                max_steps=args.max_steps,
                seed=args.seed + epoch,
            )
            print(f"  → Solve rate: {eval_metrics['solve_rate']:.1f}% ({eval_metrics['puzzles_tested']} puzzles) | Avg reward: {eval_metrics['avg_reward']:.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    print(f"\nSaved final policy to {args.output}")
    if best_model_path.exists():
        print(f"Best model saved to {best_model_path} (val_loss={best_val_loss:.6f})")

    # Final evaluation
    if args.eval_puzzles:
        print(f"\nRunning final puzzle solve evaluation...")
        # Load best model for final eval
        if best_model_path.exists():
            model = FlowQNetwork(in_channels=MAX_CHANNELS, use_dueling=args.use_dueling)
            model.load_state_dict(torch.load(best_model_path))
            model.to(device)
            print(f"  Using best model from {best_model_path}")

        eval_metrics = evaluate_solve_rate(
            model,
            args.eval_puzzles,
            device,
            max_episodes=args.eval_episodes,
            max_steps=args.max_steps,
            seed=args.seed + 9999,
        )
        print(f"  Final solve rate: {eval_metrics['solve_rate']:.1f}% ({eval_metrics['puzzles_tested']} puzzles) | Avg reward: {eval_metrics['avg_reward']:.2f}")


if __name__ == "__main__":
    main()
