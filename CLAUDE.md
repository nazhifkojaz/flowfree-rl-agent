# CLAUDE.md

Guidelines for Claude Code when working inside this repository. The project focuses on a Transformer-enhanced DQN agent for solving FlowFree puzzles, built on the env2 environment with modular reward shaping, expert demonstration mixing, and curriculum learning.

## Project Overview
- **Environment** (`rl/env/`): board config dataclasses, modular reward engines, observation/mask builders, `FlowFreeEnv`, curriculum support, and trace generation.
- **Solver** (`rl/solver/`): supervised warm-start utilities, prioritized replay buffer with expert mixing, Transformer Q-network, DQN trainer with loop detection.
- **Pipelines** (`scripts/`): shell orchestrators for dataset split, trace generation, supervised warm-start, DQN training, evaluation, and rollout visualization.

End-to-end workflow:
```
puzzle_data.csv ─→ split_dqn_dataset.py
                    ↓
           generate_traces_from_completion.py
                    ↓
         train_supervised.py  (warm-start policy)
                    ↓
           train_dqn.py  (Transformer + PER + expert buffer)
                    ↓
        holdout evaluation (inline in pipeline)
                    ↓
    render_holdout_rollouts.py  (optional visualization)
```

## Development Commands
```bash
# Install dependencies
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt

# Activate virtualenv if desired
python -m venv .venv && source .venv/bin/activate

# Run tests
pytest
```

### Trace Generation
```bash
python rl/env/generate_traces_from_completion.py \
  --csv data/puzzle_data.csv \
  --out-dir data/rl_traces/dqn_supervised \
  --max-size 6 --force
```

### Supervised Warm-start
```bash
python rl/solver/train_supervised.py \
  --traces-dir data/rl_traces/dqn_supervised \
  --output models/dqn_supervised_warmstart.pt \
  --epochs 60 --batch-size 128
```

### DQN Training (env2)
```bash
python rl/solver/train_dqn.py \
  --puzzle-csv data/dqn_train.csv \
  --validation-csv data/dqn_val.csv \
  --episodes 2000 \
  --batch-size 128 \
  --buffer-size 50000 \
  --epsilon-schedule linear \
  --epsilon-start 1.0 --epsilon-end 0.05 --epsilon-linear-steps 1500 \
  --env-backend env2 \
  --policy-init models/dqn_supervised_warmstart.pt \
  --log-root logs/dqn_runs
```

#### Advanced DQN Parameters

**Expert Demonstration Mixing:**
- `--expert-buffer-size N` - Size of expert demonstration buffer (default: 0, disabled)
- `--expert-sample-ratio R` - Fraction of batch sampled from expert buffer (default: 0.0)

**Loop Detection & Prevention:**
- `--loop-penalty P` - Penalty for revisiting board states (default: -0.15)
- `--loop-window W` - Number of recent states to track for loops (default: 6)

**Progress Shaping:**
- `--progress-bonus B` - Bonus reward per newly filled cell (default: 0.02)

**Epsilon Decay:**
- `--epsilon-schedule TYPE` - "linear" or "exp" (exponential) decay (default: linear)

**Curriculum Learning (6x6 boards):**
- `--curriculum-six-prob-start` - Initial 6x6 sampling probability (default: 0.0)
- `--curriculum-six-prob-end` - Final 6x6 sampling probability (default: 0.0)
- `--curriculum-six-prob-episodes` - Episodes over which to ramp up (default: 0)

### Full Pipeline
```bash
bash scripts/full_dqn_pipeline_env2.sh
```
Flags such as `--skip-data-split`, `--skip-traces`, `--skip-supervised`, or `--episodes N` let you shorten smoke tests.

### Holdout Evaluation & Visualization
Holdout evaluation is integrated into the pipeline script (step 5). For manual evaluation:
```bash
bash scripts/eval_holdout_manual.sh
```

To render rollout visualizations with GIFs:
```bash
python scripts/render_holdout_rollouts.py \
  --rollout-dir logs/dqn_full_pipeline_env2/holdout_rollouts/dqn_final_env2 \
  --output-dir logs/dqn_full_pipeline_env2/holdout_rollouts/dqn_final_env2/gifs \
  --save-gif --gif-duration 140
```

Pipeline flags for rollout recording:
- `--record-holdout-rollouts MODE` - Record rollouts: none/solved/unsolved/both (default: both)
- `--holdout-gif` - Enable GIF generation from rollout frames
- `--holdout-gif-duration MS` - GIF frame duration in milliseconds (default: 140)
- `--holdout-rollout-max N` - Maximum rollouts to record (0 = unlimited)

## Architecture Notes

### rl/env/
- `config.py`: `BoardShape`, `EnvConfig`, `RewardPreset`, observation specs, and default channel layouts.
- `env.py`: `FlowFreeEnv`, the Gym-like loop exposing `reset` / `step` with dict observations (`tensor`, `action_mask`, metadata).
- `state.py`: immutable board state with helper diffs and constraint checks (disconnects, dead pockets, degree violations).
- `observation.py` & `mask.py`: incremental builders for tensors and legal-action masks, including undo handling.
- `trace.py`: serialization helpers for env2 trajectories (JSON backed).
- `generate_traces_from_completion.py`: replays finished boards from `CompletePuzzle` strings to produce supervised traces.
- `curriculum.py`: `SuccessRateCurriculum` for adaptive difficulty progression based on solve rates.
- `constants.py`: environment-level constants (actions, directions, rewards).
- **`rewards/` submodule**: Modular reward engine architecture
  - `base.py`: `RewardEngine` protocol for composable reward components
  - `completion.py`: `CompletionReward` - rewards for completing/sustaining colors
  - `potential.py`: `PotentialReward` - distance-based potential shaping
  - `constraints.py`: `ConstraintPenalty` - penalties for dead pockets, disconnects, degree violations
  - `composite.py`: `CompositeReward` - combines multiple reward engines
  - `__init__.py`: `build_reward_system` factory for assembling reward pipelines

### rl/solver/
- `core/constants.py`: limits for board size, color count, channel count, and flattened action dimension.
- `core/config.py`: dataclasses for trainer configs (DQN, evaluation, reward settings).
- `core/logging.py`: console/TensorBoard/MLflow hooks (NullRunLogger by default).
- `policies/backbone.py`: `FlowBackbone`, a residual CNN with optional spatial attention used by both the supervised policy and Q-network.
- `policies/policy.py`: `FlowPolicy` for supervised warm-start; same backbone, lightweight policy head.
- `policies/q_network.py`: `FlowQNetwork`, which turns per-colour tokens into action logits via a Transformer encoder plus dueling streams.
- `data/replay.py`: replay buffer with prioritized sampling, importance weights, and expert demonstration mixing.
- `dataset.py`: `SupervisedTrajectoryDataset` for loading and batching supervised traces.
- `reward_settings.py`: `RewardSettings` dataclass for unified reward configuration.
- `train_supervised.py`: CLI trainer for imitation learning from generated traces.
- `train_dqn.py`: CLI front-end; wires configs, creates env2 instances, runs `rl.solver.trainers.dqn.run_training`.
- `trainers/dqn.py`: training loop with PER, target network sync, loop detection, expert buffer mixing, evaluation hooks, and logging utilities (1327 lines).

### Scripts
- `scripts/split_dqn_dataset.py`: split `puzzle_data.csv` into train/val/test CSVs with stratified sampling.
- `scripts/full_dqn_pipeline_env2.sh`: orchestrates split → traces → supervised warm-start → DQN training → hold-out eval (780 lines, 60+ parameters).
- `scripts/analyze_reward_breakdown.py`: analyze reward component contributions from training logs.
- `scripts/diagnose_training.py`: diagnostic utilities for training runs.
- `scripts/render_holdout_rollouts.py`: visualize holdout episodes as frame sequences or GIFs.
- `scripts/eval_holdout_manual.sh`: manual holdout evaluation script.
- `scripts/train_dqn_fixed.sh`: fixed configuration DQN training script.
- `scripts/view_tensorboard.sh`, `scripts/view_training.sh`: quick launchers for monitoring training.

## Testing
- `pytest tests/test_env2_state.py` - environment state mechanics and transitions.
- `pytest tests/test_rl_supervised.py` - supervised dataset and FlowPolicy sanity checks.
- `pytest tests/test_rl_dqn.py -k smoke` - short DQN trainer smoke test.
- `pytest tests/test_trainers_unit.py` - trainer component unit tests.
- `pytest tests/test_trainers_smoke.py` - trainer integration smoke tests.

## Additional Documentation
- `docs/pipeline_overview.md` - comprehensive architecture walkthrough (173 lines)
- `docs/pipeline_parameters.md` - parameter tuning guide for DQN training (70 lines)
- `docs/replay_buffer.md` - replay buffer implementation details (84 lines)
- `AGENTS.md` - general repository guidelines for AI coding assistants
- `GEMINI.md` - Gemini-specific instructions (similar to this file)
- `rl/env/README.md` - environment-specific documentation
- `rl/solver/REFACTORING_NOTES.md` - refactoring status and legacy component notes

## Style & Expectations
- Python 3.10+, black-compatible formatting, type hints on public APIs.
- Observation/reward constants live near the top of each module.
- Keep new data files out of version control (`data/`, `logs/`, `models/` are already gitignored).
- When adding flags, update `scripts/full_dqn_pipeline_env2.sh` so the pipeline stays reproducible.

## Key Features (Recent Additions)

### Loop Detection & Prevention
The DQN trainer now tracks recent board states and applies penalties for revisiting states, preventing the agent from getting stuck in cycles. Configurable via `--loop-penalty` and `--loop-window`.

### Expert Demonstration Mixing
The replay buffer supports mixing expert demonstrations (from the PNS solver) with agent experience. Configure via `--expert-buffer-size` and `--expert-sample-ratio` to enable DQfD-style learning.

### Progress-Based Shaping
A progress bonus rewards the agent for filling new cells, encouraging exploration and forward progress. Configurable via `--progress-bonus`.

### Rollout Visualization
The pipeline can record holdout evaluation episodes as frame sequences and optionally generate GIFs for visual analysis. Use `--record-holdout-rollouts` and `--holdout-gif` flags.

### Modular Reward System
Rewards are now composed of separate engines (`PotentialReward`, `CompletionReward`, `ConstraintPenalty`, etc.) that can be mixed and matched via `RewardPreset` or custom configurations.

### Curriculum Learning
The trainer supports gradually increasing the proportion of harder (6x6) boards during training via curriculum parameters.
