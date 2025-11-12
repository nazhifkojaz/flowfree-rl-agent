from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from rl.solver.dataset import SupervisedTrajectoryDataset, discover_trace_files
from rl.solver.constants import MAX_CHANNELS
from rl.solver.policies.policy import FlowPolicy, masked_cross_entropy, save_policy


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Supervised warm-start training for FlowFree policies")
    parser.add_argument(
        "--traces-dir",
        type=Path,
        nargs="+",
        default=[Path("data/rl_traces/dqn_supervised")],
        help="Directory or directories containing trajectory JSON traces",
    )
    parser.add_argument("--output", type=Path, default=Path("models/dqn_supervised_warmstart.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--max-traces", type=int, default=None, help="Optional cap on number of trace files to load")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Fraction of data reserved for validation")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate_batch(batch):
    states = torch.stack([ex.state for ex in batch])
    masks = torch.stack([ex.action_mask for ex in batch]).to(torch.bool)
    actions = torch.tensor([ex.action for ex in batch], dtype=torch.long)
    return states, masks, actions


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    seed_everything(args.seed)

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trace_files: list[Path] = []
    for root in args.traces_dir:
        trace_files.extend(discover_trace_files(root))

    if not trace_files:
        raise SystemExit("No trajectory files found; generate traces before running supervised training.")

    trace_files = sorted(trace_files)
    if args.max_traces is not None:
        trace_files = trace_files[: args.max_traces]

    dataset = SupervisedTrajectoryDataset(trace_files)
    if len(dataset) == 0:
        raise SystemExit("Loaded dataset is empty; verify trace contents.")

    val_size = int(len(dataset) * max(0.0, min(1.0, args.val_ratio)))
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed))
    else:
        train_dataset, val_dataset = dataset, None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = (
        DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_batch)
        if val_dataset is not None
        else None
    )

    model = FlowPolicy(in_channels=MAX_CHANNELS).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Loaded {len(trace_files)} traces → {len(dataset)} examples")
    if val_loader is not None:
        print(f"Train examples: {train_size} | Validation examples: {val_size}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for states, masks, actions in train_loader:
            states = states.to(device)
            masks = masks.to(device)
            actions = actions.to(device)

            logits = model(states, masks)
            loss = masked_cross_entropy(logits, actions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item())
            batches += 1

        train_loss = epoch_loss / max(1, batches)

        val_loss = None
        if val_loader is not None:
            model.eval()
            total_val = 0.0
            with torch.no_grad():
                for states, masks, actions in val_loader:
                    states = states.to(device)
                    masks = masks.to(device)
                    actions = actions.to(device)
                    logits = model(states, masks)
                    loss = masked_cross_entropy(logits, actions)
                    total_val += float(loss.item())
            val_loss = total_val / max(1, len(val_loader))

        if val_loss is not None:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")
        else:
            print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}")

    save_policy(model.cpu(), args.output)
    print(f"Saved supervised policy to {args.output}")


if __name__ == "__main__":
    main()
