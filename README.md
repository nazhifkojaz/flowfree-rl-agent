# FlowFree RL Agent

A Deep Q-Network (DQN) agent enhanced with Transformer architecture for solving FlowFree puzzles. This project implements advanced techniques such as Prioritized Experience Replay (PER), expert demonstration mixing, loop detection, and a modular reward system to achieve state-of-the-art performance on FlowFree puzzles.

## 🎯 Performance Highlights

- **Current Results:** 50.57% solve rate on 5×5 holdout set (176 puzzles)
- **Latest Model:** `dqn_20251111-185706.pt` (training run: dqn_20251111-120251)
- **Key Improvements:** Loop detection (-0.5 penalty, window=6), undo penalty (-0.25), progress bonus (+0.02)

## 📊 Key Features

### Neural Architecture
- **Transformer-Enhanced DQN**: Per-color token construction with 2-layer Transformer encoder (4 attention heads) for relational reasoning
- **Dueling Architecture**: Separate value and advantage streams for improved credit assignment
- **Residual CNN Backbone**: 6-block ResNet with GroupNorm and spatial attention (128 channels, ~1M parameters)
- **Action Masking**: Incremental mask updates for efficient legal action filtering

### Training Infrastructure
- **Prioritized Experience Replay (PER)**: α=0.6, β annealing from 0.4→1.0 for efficient sampling
- **Expert Demonstration Mixing (DQfD)**: 5000-capacity expert buffer, 25% sampling ratio during early training
- **Loop Detection**: Position oscillation tracking (window=6) with -0.5 penalty, 5× multiplier for undo-redo patterns
- **Progress-Based Shaping**: +0.02 bonus per newly filled cell to encourage forward exploration

### Modular Reward System
Composable reward engines for rapid experimentation:
- **PotentialReward**: Distance-based potential shaping (scale: 0.35)
- **CompletionReward**: Bonuses for completing colors (+1.8), sustaining completions (+0.1)
- **ConstraintPenalty**: Penalties for disconnects (-0.06), degree violations (-0.08)
- **LoopPenalty**: Prevents cyclic behavior (-0.5 base, -2.5 for undo-redo oscillations)
- **UndoPenalty**: Discourages excessive backtracking (-0.25 per undo)

## 🏗️ Architecture Overview

### End-to-End Pipeline

```
puzzle_data.csv (3815 puzzles, 3×3 to 6×6)
    │
    ├─ [Stage 1] Dataset Split (stratified by size & color count, uses puzzle_data.csv)
    │   → data/dqn_train.csv (70%)
    │   → data/dqn_val.csv (15%)
    │   → data/dqn_test.csv (15%)
    │
    ├─ [Stage 2] Trace Generation (DFS reconstruction from solved boards)
    │   → data/rl_traces/*.json
    │   Reconstructs state-action trajectories for supervised warm-start
    │
    ├─ [Stage 3] Supervised Warm-Start (behavioral cloning)
    │   → models/dqn_supervised_warmstart.pt
    │   Trains FlowPolicy via masked cross-entropy (10 epochs)
    │
    ├─ [Stage 4] DQN Training (Transformer + PER + expert mixing)
    │   → logs/dqn_full_pipeline_env2/dqn_<timestamp>/
    │   → models/dqn_<timestamp>.pt
    │   3000 episodes, ε: 1.0→0.05 linear decay, batch=64
    │
    └─ [Stage 5] Holdout Evaluation (ε=0 deterministic policy)
        → Solve rate, avg reward, constraint violation analysis
        → Optional: GIF rollouts for qualitative analysis
```

**Orchestration:** `scripts/full_dqn_pipeline_env2.sh` (780 lines, 60+ configurable parameters)

### Project Structure

```
flowfree-rl-agent/
├── rl/
│   ├── env/                    # FlowFree environment (gym-compatible)
│   │   ├── config.py           # BoardShape, EnvConfig, ObservationSpec, RewardPreset
│   │   ├── constants.py        # Action constants, directions, empty cell marker
│   │   ├── curriculum.py       # SuccessRateCurriculum for adaptive difficulty
│   │   ├── env.py              # Main environment loop (reset/step/render)
│   │   ├── mask.py             # Incremental action masking
│   │   ├── observation.py      # Multi-channel state representation (39 channels)
│   │   ├── rewards/            # Modular reward engines
│   │   │   ├── base.py         # RewardEngine protocol
│   │   │   ├── potential.py    # Distance-based potential shaping
│   │   │   ├── completion.py   # Color completion bonuses
│   │   │   ├── constraints.py  # Dead pocket/disconnect/degree penalties
│   │   │   └── composite.py    # CompositeReward aggregator
│   │   ├── state.py            # Immutable BoardState with StateDiff transitions
│   │   ├── trace.py            # Trajectory serialization
│   │   ├── utils.py            # Utility functions (string_to_tokens)
│   │   └── generate_traces_from_completion.py
│   │
│   └── solver/                 # DQN training infrastructure
│       ├── core/               # Configuration dataclasses
│       │   ├── config.py       # DQNTrainingConfig, EvaluationConfig
│       │   ├── constants.py    # MAX_SIZE, MAX_COLORS, action space limits
│       │   └── logging.py      # Console/TensorBoard/MLflow hooks
│       ├── data/
│       │   └── replay.py       # PER with expert buffer mixing
│       ├── policies/
│       │   ├── backbone.py     # FlowBackbone (ResNet + spatial attention)
│       │   ├── policy.py       # FlowPolicy (supervised warm-start)
│       │   └── q_network.py    # FlowQNetwork (Transformer DQN)
│       ├── trainers/
│       │   └── dqn.py          # Main training loop
│       ├── constants.py
│       ├── reward_settings.py  # RewardSettings dataclass
│       ├── dataset.py          # SupervisedTrajectoryDataset
│       ├── evaluate_holdout.py # Standalone evaluation script
│       ├── train_supervised.py # Behavioral cloning CLI
│       └── train_dqn.py        # DQN training CLI
│
├── scripts/                    # Training pipelines & utilities
│   ├── full_dqn_pipeline_env2.sh  # Full end-to-end pipeline
│   ├── split_dqn_dataset.py       # Stratified train/val/test split
│   ├── train_dqn_improved.sh      # Phase 1+2 recommended settings
│   ├── eval_holdout_manual.sh     # Manual holdout evaluation
│   ├── render_holdout_rollouts.py # Generate GIFs from eval rollouts
│   └── view_tensorboard.sh
└── data/                       # Puzzle datasets (please do generate RL traces by yourself)
```

## 🚀 Quick Start

### Installation

```bash
# Create conda environment and install uv
conda create -n flowfree-rl python=3.10
conda activate flowfree-rl
pip install uv

# Install dependencies with uv
uv pip install -r requirements.txt
uv pip install -r requirements-dev.txt
```

### Run Full Training Pipeline

```bash
bash scripts/full_dqn_pipeline_env2.sh
```

### Evaluate Trained Model

```bash
# Evaluate on holdout set with rollout recording
bash scripts/eval_holdout_manual.sh \
    models/dqn_20251111-185706.pt \
    both \
    logs/eval_rollouts \
    0 \
    false

# Arguments: <model_path> <record_mode> <output_dir> <max_rollouts> <generate_gifs>
```

## 🎮 Environment Details

### FlowFree Puzzle Rules

Connect matching colored endpoints on a grid, filling all cells without crossing paths.

**Simplified Rules (Single-Endpoint Growth):**
- Each color has one active head (grows from one endpoint at a time)
- Actions: UP, RIGHT, DOWN, LEFT, UNDO (5 per color)
- Episode terminates when solved or max_steps reached

### Observation Space

**Multi-Channel Tensor:** `(batch, 39, height, width)` with channel-first convention

**Channels (configurable via `env2_channels`):**
1. **Occupancy** (12 channels): Per-color one-hot occupancy
2. **Endpoints** (12 channels): Binary mask of start/target positions
3. **Heads** (12 channels): Current head position per color
4. **Free cells** (1 channel): Unoccupied cells
5. **Congestion** (1 channel): Density of occupied neighbors (normalized)
6. **Distance** (1 channel): Manhattan distance to nearest endpoint

**Total:** 12×3 + 3 = **39 channels** (adjustable for 12 colors max, scales for fewer)

### Action Space

- **Discrete:** `num_colors × 5 = 60 actions` (for 12 colors max)
- **Flattened Encoding:** `action_idx = color * 5 + direction`
- **Masking:** Invalid actions (out-of-bounds, collisions, completed colors) masked to -inf

### Reward Function

**Current Configuration (from 50.57% run):**
```python
# Terminal rewards
solve_bonus: 35.0                    # Puzzle solved
constraint_free_bonus: 5.0           # Solved with zero violations
unsolved_penalty: -2.0               # Truncated without solution

# Per-step rewards
move_penalty: -0.05                  # Each action
distance_bonus: 0.35                 # Potential shaping scale
complete_color_bonus: 1.8            # Color reaches target
complete_sustain_bonus: 0.1          # Per step color stays complete
complete_revert_penalty: 2.0         # Undo completed color

# Constraint penalties
disconnect_penalty: -0.06            # Per disconnected cell
degree_penalty: -0.08                # Per degree violation (>2 neighbors)
dead_pocket_penalty: 0.0             # Disabled (included in disconnects)

# Behavioral penalties
undo_penalty: -0.25                  # Discourage excessive backtracking
loop_penalty: -0.5                   # Revisiting states (6-step window)
invalid_penalty: -0.3                # Illegal actions (shouldn't happen)

# Exploration bonus
progress_bonus: 0.02                 # Per newly filled cell
```

## 🔧 Training Configuration

### Current Hyperparameters (Current Best: 50.57%)

```python
# Architecture
backbone_hidden: 128
backbone_blocks: 6
transformer_layers: 2
transformer_heads: 4
use_dueling: True              # Dueling DQN architecture

# Training
episodes: 3000                 # Can increase this but also adjust epsilon decay
batch_size: 64                 # Limited to 64 due to my GPU, can be increased
buffer_size: 60000             # Increasing might help or hurt (polluted by failed replays) 
lr: 1e-4 (Adam)
gamma: 0.99
target_update: 500             # Hard update every 500 steps
grad_clip: 0.5
epsilon_schedule: "linear"
epsilon_start: 1.0
epsilon_end: 0.05
epsilon_linear_steps: 3000

# Prioritized Experience Replay
per_alpha: 0.6                 # Priority exponent
per_beta: 0.4                  # IS correction (anneals to 1.0)
per_beta_increment: 1e-4

# Expert Demonstration Mixing (Experimental, might not need this later)
expert_buffer_size: 5000
expert_sample_ratio: 0.25      # 25% of batch from expert buffer

# Loop Detection & Prevention
loop_penalty: -0.5             # Base penalty for revisiting states
loop_window: 6                 # Track last 6 states (higher = more sensitive)
undo_penalty: -0.25            # Discourage excessive backtracking
progress_bonus: 0.02           # Encourage forward progress

# Key Reward Parameters (see Reward Function for full list)
solve_bonus: 35.0
unsolved_penalty: -2.0
disconnect_penalty: -0.06
degree_penalty: -0.08
```

### Known Failure Modes

**Common failure patterns (from training logs):**

1. **Disconnect Cascades (45%):** Agent extends one color, creating isolated cells that other colors can't reach
2. **Degree Violations (35%):** Creates cells with 3+ occupied neighbors (invalid grid patterns)
3. **Premature Color Completion (15%):** Completes easy colors first, getting trapped later
4. **Exploration Exhaustion (5%):** Gets stuck in local optima late in training (ε < 0.1)


## 🔬 Advanced Usage

### Custom Training Configuration

```bash
python rl/solver/train_dqn.py \
    --puzzle-csv data/dqn_train.csv \
    --validation-csv data/dqn_val.csv \
    --episodes 3000 \
    --batch-size 64 \
    --buffer-size 60000 \
    --use-dueling \
    --expert-buffer-size 5000 \
    --expert-sample-ratio 0.25 \
    --loop-penalty -0.5 \
    --loop-window 6 \
    --env2-undo-penalty -0.25 \
    --progress-bonus 0.02 \
    --disconnect-penalty -0.06 \
    --degree-penalty -0.08 \
    --policy-init models/dqn_supervised_warmstart.pt \
    --log-root logs/custom_run
```

### Pipeline Flags

**Key parameters for `scripts/full_dqn_pipeline_env2.sh`:**

```bash
--skip-data-split              # Skip stage 1 (use existing split)
--skip-traces                  # Skip stage 2 (use existing traces)
--skip-supervised              # Skip stage 3 (no warm-start)
--episodes 3000                # DQN training episodes
--buffer-size 60000            # Replay buffer capacity
--use-dueling                  # Enable dueling DQN architecture
--expert-buffer-size 5000      # Expert buffer capacity
--expert-sample-ratio 0.25     # Fraction of batch from expert buffer
--loop-penalty -0.5            # Base loop penalty
--loop-window 6                # Track last N states for loops
--env2-undo-penalty -0.25      # Undo action penalty
--progress-bonus 0.02          # Bonus per newly filled cell
--disconnect-penalty -0.06     # Penalty per disconnected cell
--degree-penalty -0.08         # Penalty per degree violation
--record-holdout-rollouts both # Record solved/unsolved/both
--holdout-gif                  # Generate GIF visualizations
```

## 🔗 References

TBD

## 🔗 Related Projects

For a classical (non-RL) solvers I developed for FlowFree, check out [Flowfree-Solver](https://github.com/nazhifkojaz/Flowfree-Solver)!

## 🤝 Contributing

This project uses AI assistant guidelines documented in:
- **CLAUDE.md**: Claude-specific development instructions

**Status:** Work-in-progress • 50.57% solve rate on 5×5 boards
