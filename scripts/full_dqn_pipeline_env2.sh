#!/bin/bash
################################################################################
# Full DQN Training Pipeline (Env2 Backend)
#
# Runs the complete workflow with FlowFreeEnv for RL training/evaluation:
# 1. Data split (train/val/test)
# 2. Trace generation for supervised warm-start
# 3. Supervised policy pre-training
# 4. DQN training using env2
# 5. Hold-out evaluation (env2)
################################################################################

set -e
set -u
set -o pipefail

# Styling
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Defaults (tune as needed)
RUN_ID="dqn_$(date +%Y%m%d-%H%M%S)"
EPISODES=4000
BATCH_SIZE=192
BUFFER_SIZE=80000
TRACE_LIMIT=3000
MIN_SIZE=5
MAX_SIZE=5
MAX_COLORS=10
SEED=42
EVAL_INTERVAL=100
EVAL_EPISODES=20
LOG_EVERY=25
ENV2_CHANNELS=("occupancy" "endpoints" "heads" "free" "congestion" "distance")
ENV2_REWARD="potential"
COMPLETE_BONUS=2.0
SOLVE_BONUS=40.0
SOLVE_EFFICIENCY_BONUS=0.6
EPSILON_START=1.0
EPSILON_END=0.10
EPSILON_DECAY=20000
CONSTRAINT_FREE_BONUS=0.0  # DISABLED
PENALTY_WARMUP=800
DISCONNECT_PENALTY=-0.06
DEGREE_PENALTY=-0.08
INVALID_PENALTY=-0.3
COMPLETE_SUSTAIN_BONUS=0.0  # DISABLED
COMPLETE_REVERT_PENALTY=0.0  # DISABLED (was 2.0)
UNDO_PENALTY=-0.12
EPSILON_SCHEDULE="linear"
EPSILON_LINEAR_STEPS="4000"
UNSOLVED_PENALTY=-4.0
UNSOLVED_PENALTY_START=0.0
UNSOLVED_PENALTY_WARMUP=500
CURRICULUM_SIX_START=0.0
CURRICULUM_SIX_END=0.0
CURRICULUM_SIX_EPISODES=0
LOOP_PENALTY=-1.8
LOOP_WINDOW=6
PROGRESS_BONUS=0.015
STEPS_PER_EPISODE=""
EXPERT_BUFFER_SIZE=5000
EXPERT_SAMPLE_RATIO=0.20
USE_AMP=true
GRADIENT_ACCUMULATION_STEPS=2
DISABLE_TENSORBOARD=false
HOLDOUT_ROLLOUT_MODE="both"
HOLDOUT_GIF=true
HOLDOUT_GIF_DURATION=140
HOLDOUT_ROLLOUT_MAX=0
TRACE_COMPLETION_MODES=("normal" "longest" "blocked")
TRACE_MODE_OVERRIDE=false
TRACE_VARIANTS=1
TRACE_SHUFFLE=false

LOG_DIR="logs"
RUN_DIR="${LOG_DIR}/${RUN_ID}"
HOLDOUT_ROLLOUT_DIR="${RUN_DIR}/holdout_rollouts"

# If no explicit step cap provided, default to 2 * board area
if [ -z "$STEPS_PER_EPISODE" ]; then
    STEPS_PER_EPISODE=$((MAX_SIZE * MAX_SIZE * 2))
fi

# Supervised config
SUPERVISED_EPOCHS=10
SUPERVISED_BATCH_SIZE=64
SUPERVISED_LR=0.0001

# Flags
SKIP_DATA_SPLIT=false
SKIP_TRACES=false
SKIP_SUPERVISED=false
USE_SIMPLE_REWARDS=false
QUICK_MODE=false

show_usage() {
    cat <<EOF
Usage: $0 [options]
  --skip-data-split      Skip dataset splitting
  --skip-traces          Skip trace generation
  --skip-supervised      Skip supervised warm start
  --simple-rewards       Use potential-based reward preset
  --quick                Fast settings (50 episodes, fewer epochs)
  --episodes N           Override episode count (default: 4000)
  --batch-size N         Override DQN batch size (default: 64)
  --buffer-size N        Override replay buffer size (default: 80000)
  --trace-limit N        Limit number of traces generated (default: 3000)
  --trace-mode MODE      Completion mode for trace generation (repeatable; default: normal)
  --trace-variants N     Number of variants per puzzle in normal mode (default: 1)
  --trace-shuffle        Shuffle color order variants in normal mode
  --supervised-epochs N  Override supervised epochs (default: 10)
  --supervised-batch-size N
  --supervised-lr LR     Override supervised learning rate (default: 1e-4)
  --epsilon-start VAL    Starting epsilon for DQN (default: 1.0)
  --epsilon-end VAL      Ending epsilon for DQN (default: 0.10)
  --complete-bonus VAL   Completion reward bonus (default: 1.8)
  --solve-bonus VAL      Solve reward bonus (default: 35.0)
  --solve-efficiency-bonus VAL  Bonus per step remaining when solved (default: 0.5)
  --constraint-free-bonus VAL  Bonus when no constraints are violated (default: 0.0, DISABLED)
  --penalty-warmup N     Episodes over which to ramp constraint penalties (default: 600)
  --disconnect-penalty VAL  Disconnect penalty (default: -0.06)
  --degree-penalty VAL      Degree penalty (default: -0.08)
  --invalid-penalty VAL     Invalid move penalty (default: -0.3)
  --undo-penalty VAL        Undo action penalty (default: -0.1)
  --complete-sustain-bonus VAL  Sustain bonus per completed colour (default: 0.1)
  --complete-revert-penalty VAL Penalty when undo reopens a colour (default: 4.0)
  --unsolved-penalty VAL    Terminal penalty for unsolved boards (default: -2.0)
  --unsolved-penalty-start VAL  Starting unsolved penalty before warmup (default: 0.0)
  --unsolved-penalty-warmup N   Episodes to ramp unsolved penalty (default: 500)
  --loop-penalty VAL       Penalty applied when board repeats within loop window (default: -2.0)
  --loop-window N          Number of recent board states to track for loop penalty (default: 10)
  --progress-bonus VAL     Bonus per newly filled cell (default: 0.02)
  --epsilon-schedule TYPE   Exploration schedule (linear or exp, default: linear)
  --epsilon-linear-steps N  Episodes to decay epsilon when using linear schedule (default: --episodes)
  --curriculum-six-start VAL   Initial probability of sampling 6x6 boards (default: 0.0)
  --curriculum-six-end VAL     Final probability of sampling 6x6 boards (default: 0.0)
  --curriculum-six-episodes N  Episodes over which to anneal the curriculum probability (default: 0)
  --eval-interval N      Evaluation cadence (default: 100)
  --eval-episodes N      Evaluation rollouts per checkpoint (default: 20)
  --log-every N          Training print frequency (default: 25)
  --env2-reward NAME     Reward preset name for env2 (default: potential)
  --env2-channels LIST   Space-separated env2 observation channels
  --steps-per-episode N  Override max steps per episode (default: auto = area + 12)
  --record-holdout-rollouts MODE  Record holdout rollouts: none/solved/unsolved/both (default: none)
  --holdout-rollout-dir PATH      Directory for holdout rollout dumps
  --holdout-rollout-max N         Limit number of holdout rollouts saved (0 = unlimited)
  --holdout-gif                   Enable GIF generation for recorded holdout rollouts
  --holdout-gif-duration MS       GIF frame duration (default: 140)
  --expert-buffer-size N      Size of expert replay buffer (default: 0 = disabled)
  --expert-sample-ratio VAL   Fraction of batch sampled from expert buffer (default: 0.0)
  --use-amp                   Enable Automatic Mixed Precision (2-3x faster on T4/A100, default: false)
  --gradient-accumulation-steps N  Number of gradient accumulation steps (effective batch = batch * N, default: 1)
  --disable-tensorboard       Disable TensorBoard logging to save ~1-2GB RAM (CSV metrics still logged, default: false)
EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data-split) SKIP_DATA_SPLIT=true ;;
        --skip-traces) SKIP_TRACES=true ;;
        --skip-supervised) SKIP_SUPERVISED=true ;;
        --simple-rewards) USE_SIMPLE_REWARDS=true ;;
        --quick)
            QUICK_MODE=true
            EPISODES=10
            EVAL_INTERVAL=10
            SUPERVISED_EPOCHS=2
            ;;
        --episodes)
            EPISODES="$2"
            shift
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift
            ;;
        --buffer-size)
            BUFFER_SIZE="$2"
            shift
            ;;
        --trace-limit)
            TRACE_LIMIT="$2"
            shift
            ;;
        --trace-mode)
            IFS=',' read -ra NEW_TRACE_MODES <<< "$2"
            if [ "$TRACE_MODE_OVERRIDE" = false ]; then
                TRACE_COMPLETION_MODES=()
                TRACE_MODE_OVERRIDE=true
            fi
            for mode in "${NEW_TRACE_MODES[@]}"; do
                TRACE_COMPLETION_MODES+=("$mode")
            done
            shift
            ;;
        --trace-variants)
            TRACE_VARIANTS="$2"
            shift
            ;;
        --trace-shuffle)
            TRACE_SHUFFLE=true
            ;;
        --supervised-epochs)
            SUPERVISED_EPOCHS="$2"
            shift
            ;;
        --supervised-batch-size)
            SUPERVISED_BATCH_SIZE="$2"
            shift
            ;;
        --supervised-lr)
            SUPERVISED_LR="$2"
            shift
            ;;
        --epsilon-start)
            EPSILON_START="$2"
            shift
            ;;
        --epsilon-end)
            EPSILON_END="$2"
            shift
            ;;
        --epsilon-schedule)
            EPSILON_SCHEDULE="$2"
            shift
            ;;
        --epsilon-linear-steps)
            EPSILON_LINEAR_STEPS="$2"
            shift
            ;;
        --curriculum-six-start)
            CURRICULUM_SIX_START="$2"
            shift
            ;;
        --curriculum-six-end)
            CURRICULUM_SIX_END="$2"
            shift
            ;;
        --curriculum-six-episodes)
            CURRICULUM_SIX_EPISODES="$2"
            shift
            ;;
        --steps-per-episode)
            STEPS_PER_EPISODE="$2"
            shift
            ;;
        --complete-bonus)
            COMPLETE_BONUS="$2"
            shift
            ;;
        --solve-bonus)
            SOLVE_BONUS="$2"
            shift
            ;;
        --solve-efficiency-bonus)
            SOLVE_EFFICIENCY_BONUS="$2"
            shift
            ;;
        --constraint-free-bonus)
            CONSTRAINT_FREE_BONUS="$2"
            shift
            ;;
        --penalty-warmup)
            PENALTY_WARMUP="$2"
            shift
            ;;
        --disconnect-penalty)
            DISCONNECT_PENALTY="$2"
            shift
            ;;
        --degree-penalty)
            DEGREE_PENALTY="$2"
            shift
            ;;
        --invalid-penalty)
            INVALID_PENALTY="$2"
            shift
            ;;
        --undo-penalty|--env2-undo-penalty)
            UNDO_PENALTY="$2"
            shift
            ;;
        --complete-sustain-bonus)
            COMPLETE_SUSTAIN_BONUS="$2"
            shift
            ;;
        --complete-revert-penalty)
            COMPLETE_REVERT_PENALTY="$2"
            shift
            ;;
        --record-holdout-rollouts)
            HOLDOUT_ROLLOUT_MODE="$2"
            shift
            ;;
        --holdout-rollout-dir)
            HOLDOUT_ROLLOUT_DIR="$2"
            shift
            ;;
        --holdout-rollout-max)
            HOLDOUT_ROLLOUT_MAX="$2"
            shift
            ;;
        --holdout-gif)
            HOLDOUT_GIF=true
            ;;
        --holdout-gif-duration)
            HOLDOUT_GIF_DURATION="$2"
            shift
            ;;
        --expert-buffer-size)
            EXPERT_BUFFER_SIZE="$2"
            shift
            ;;
        --expert-sample-ratio)
            EXPERT_SAMPLE_RATIO="$2"
            shift
            ;;
        --use-amp)
            USE_AMP=true
            ;;
        --gradient-accumulation-steps)
            GRADIENT_ACCUMULATION_STEPS="$2"
            shift
            ;;
        --disable-tensorboard)
            DISABLE_TENSORBOARD=true
            ;;
        --unsolved-penalty)
            UNSOLVED_PENALTY="$2"
            shift
            ;;
        --unsolved-penalty-start)
            UNSOLVED_PENALTY_START="$2"
            shift
            ;;
        --unsolved-penalty-warmup)
            UNSOLVED_PENALTY_WARMUP="$2"
            shift
            ;;
        --loop-penalty)
            LOOP_PENALTY="$2"
            shift
            ;;
        --loop-window)
            LOOP_WINDOW="$2"
            shift
            ;;
        --progress-bonus)
            PROGRESS_BONUS="$2"
            shift
            ;;
        --eval-interval)
            EVAL_INTERVAL="$2"
            shift
            ;;
        --eval-episodes)
            EVAL_EPISODES="$2"
            shift
            ;;
        --log-every)
            LOG_EVERY="$2"
            shift
            ;;
        --env2-reward)
            ENV2_REWARD="$2"
            shift
            ;;
        --env2-channels)
            shift
            ENV2_CHANNELS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do
                ENV2_CHANNELS+=("$1")
                shift
            done
            (( $# > 0 )) && continue
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
    shift
done

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║          Full DQN Training Pipeline (Env2 Backend)         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Configuration:"
echo "  Run ID:          $RUN_ID"
echo "  Episodes:         $EPISODES"
echo "  Batch size:       $BATCH_SIZE"
echo "  Buffer size:      $BUFFER_SIZE"
echo "  Puzzle size:      ${MIN_SIZE}×${MIN_SIZE} to ${MAX_SIZE}×${MAX_SIZE}"
echo "  Max colors:       $MAX_COLORS"
echo "  Seed:             $SEED"
echo "  Env2 reward:      $ENV2_REWARD"
echo "  Env2 channels:    ${ENV2_CHANNELS[*]}"
echo "  Epsilon start:    $EPSILON_START"
echo "  Epsilon End:      $EPSILON_END"
echo "  Epsilon Decay:    $EPSILON_DECAY"
echo "  Disconnect pen.:  $DISCONNECT_PENALTY"
echo "  Degree pen.:      $DEGREE_PENALTY"
echo "  Invalid pen.:     $INVALID_PENALTY"
echo "  Complete bonus:   $COMPLETE_BONUS"
echo "  Sustain bonus:    $COMPLETE_SUSTAIN_BONUS"
echo "  Revert penalty:   $COMPLETE_REVERT_PENALTY"
echo "  Solve bonus:      $SOLVE_BONUS"
echo "  Unsolved penalty: $UNSOLVED_PENALTY (start $UNSOLVED_PENALTY_START, warmup $UNSOLVED_PENALTY_WARMUP)"
echo "  Loop penalty:     $LOOP_PENALTY (window $LOOP_WINDOW)"
echo "  Progress bonus:   $PROGRESS_BONUS"
echo "  Expert buffer:    size=$EXPERT_BUFFER_SIZE ratio=$EXPERT_SAMPLE_RATIO"
echo "  Curriculum 6x6:   start=$CURRICULUM_SIX_START end=$CURRICULUM_SIX_END over $CURRICULUM_SIX_EPISODES eps"
if [ -n "$STEPS_PER_EPISODE" ]; then
    echo "  Steps/episode:    $STEPS_PER_EPISODE"
else
    echo "  Steps/episode:    auto (area + 12)"
fi
echo "  Simple rewards:   $USE_SIMPLE_REWARDS"
echo "  Quick mode:       $QUICK_MODE"
echo ""

################################################################################
# Step 1: Data Split
################################################################################
if [ "$SKIP_DATA_SPLIT" = false ]; then
    echo -e "${CYAN}Step 1/5: Splitting dataset${NC}"
    if [ -f "data/dqn_train.csv" ] && [ -f "data/dqn_val.csv" ] && [ -f "data/dqn_test.csv" ]; then
        echo -e "${YELLOW}Data splits already exist. Skipping...${NC}"
    else
        python scripts/split_dqn_dataset.py
        echo -e "${GREEN}✓ Data split completed${NC}"
    fi
else
    echo -e "${YELLOW}Skipping data split (--skip-data-split)${NC}"
fi
echo ""

################################################################################
# Step 2: Trace Generation
################################################################################
TRACE_DIR="data/"
if [ "$SKIP_TRACES" = false ]; then
    echo -e "${CYAN}Step 2/5: Generating traces${NC}"
    if [ -d "$TRACE_DIR" ] && [ "$(ls -A "$TRACE_DIR" 2>/dev/null | wc -l)" -gt 100 ]; then
        echo -e "${YELLOW}Traces already exist in $TRACE_DIR. Skipping...${NC}"
    else
        if [ "${#TRACE_COMPLETION_MODES[@]}" -eq 0 ]; then
            TRACE_COMPLETION_MODES=("normal")
        fi
        TRACE_MODE_ARGS=()
        for mode in "${TRACE_COMPLETION_MODES[@]}"; do
            TRACE_MODE_ARGS+=(--completion-mode "$mode")
        done

        TRACE_CMD=(python -m rl.env.trace_generation.cli
            --csv data/dqn_train.csv
            --out-dir "$TRACE_DIR"
            --max-size "$MAX_SIZE"
            --max-colors "$MAX_COLORS"
            --limit "$TRACE_LIMIT"
            --variants "$TRACE_VARIANTS"
        )
        if [ "$TRACE_SHUFFLE" = true ]; then
            TRACE_CMD+=(--shuffle-colors)
        fi
        TRACE_CMD+=("${TRACE_MODE_ARGS[@]}")

        "${TRACE_CMD[@]}"
        echo -e "${GREEN}✓ Traces generated${NC}"
    fi
else
    echo -e "${YELLOW}Skipping trace generation (--skip-traces)${NC}"
fi
echo ""

################################################################################
# Step 3: Supervised Pre-training
################################################################################
SUPERVISED_MODEL="models/dqn_supervised_warmstart.pt"
SUPERVISED_SCRIPT="rl/solver/train_supervised.py"
if [ "$SKIP_SUPERVISED" = false ]; then
    echo -e "${CYAN}Step 3/5: Supervised warm-start${NC}"
    if [ ! -f "$SUPERVISED_SCRIPT" ]; then
        echo -e "${YELLOW}Supervised trainer '$SUPERVISED_SCRIPT' not found. Skipping.${NC}"
        SKIP_SUPERVISED=true
    elif [ ! -d "$TRACE_DIR" ] || [ "$(ls -A "$TRACE_DIR" 2>/dev/null | wc -l)" -eq 0 ]; then
        echo -e "${YELLOW}No traces found. Skipping supervised training.${NC}"
        SKIP_SUPERVISED=true
    else
        python "$SUPERVISED_SCRIPT" \
            --traces-dir "$TRACE_DIR" \
            --output "$SUPERVISED_MODEL" \
            --epochs $SUPERVISED_EPOCHS \
            --batch-size $SUPERVISED_BATCH_SIZE \
            --lr $SUPERVISED_LR \
            --device cuda \
            --seed $SEED
            # --max-colors $MAX_COLORS \
        echo -e "${GREEN}✓ Supervised training completed${NC}"
    fi
else
    echo -e "${YELLOW}Skipping supervised warm-start (--skip-supervised)${NC}"
fi

################################################################################
# Step 4: DQN (Env2)
################################################################################
echo -e "${CYAN}Step 4/5: DQN training (env2 backend)${NC}"

mkdir -p models
mkdir -p "$RUN_DIR"
DQN_MODEL="models/${RUN_ID}.pt"

# Build env2 channel args
ENV2_CHANNEL_ARGS=()
if [ "${#ENV2_CHANNELS[@]}" -gt 0 ]; then
    ENV2_CHANNEL_ARGS=(--env2-channels "${ENV2_CHANNELS[@]}")
fi
EPS_LINEAR_ARGS=()
if [ -n "$EPSILON_LINEAR_STEPS" ]; then
    EPS_LINEAR_ARGS=(--epsilon-linear-steps "$EPSILON_LINEAR_STEPS")
fi

DQN_CMD=(python rl/solver/train_dqn.py
    --puzzle-csv data/dqn_train.csv
    --validation-csv data/dqn_val.csv
    --episodes "$EPISODES"
    --batch-size "$BATCH_SIZE"
    --buffer-size "$BUFFER_SIZE"
    --min-size "$MIN_SIZE"
    --max-size "$MAX_SIZE"
    --max-colors "$MAX_COLORS"
    --use-per
    --eval-interval "$EVAL_INTERVAL"
    --eval-episodes "$EVAL_EPISODES"
    --log-every "$LOG_EVERY"
    --output "$DQN_MODEL"
    --log-root "$LOG_DIR"
    --log-dir "$RUN_DIR"
    --seed "$SEED"
    --env-backend env2
    --env2-reward "$ENV2_REWARD"
    --use-dueling
    --epsilon-schedule "$EPSILON_SCHEDULE"
    --env2-undo-penalty "$UNDO_PENALTY"
    --epsilon-start "$EPSILON_START"
    --epsilon-end "$EPSILON_END"
    --epsilon-decay "$EPSILON_DECAY"
    --lr 1e-4
    --move-penalty -0.05
    --distance-bonus 0.35
    --invalid-penalty "$INVALID_PENALTY"
    --unsolved-penalty "$UNSOLVED_PENALTY"
    --unsolved-penalty-start "$UNSOLVED_PENALTY_START"
    --unsolved-penalty-warmup "$UNSOLVED_PENALTY_WARMUP"
    --reward-scale 1.0
    --reward-clamp 5.0
    --disconnect-penalty "$DISCONNECT_PENALTY"
    --degree-penalty "$DEGREE_PENALTY"
    --dead-end-penalty 0.0
    --segment-connection-bonus 0.0
    --path-extension-bonus 0.0
    --move-reduction-bonus 0.0
    --complete-bonus "$COMPLETE_BONUS"
    --complete-sustain-bonus "$COMPLETE_SUSTAIN_BONUS"
    --complete-revert-penalty "$COMPLETE_REVERT_PENALTY"
    --solve-bonus "$SOLVE_BONUS"
    --solve-efficiency-bonus "$SOLVE_EFFICIENCY_BONUS"
    --constraint-free-bonus "$CONSTRAINT_FREE_BONUS"
    --penalty-warmup "$PENALTY_WARMUP"
    --loop-penalty "$LOOP_PENALTY"
    --loop-window "$LOOP_WINDOW"
    --progress-bonus "$PROGRESS_BONUS"
    --expert-buffer-size "$EXPERT_BUFFER_SIZE"
    --expert-sample-ratio "$EXPERT_SAMPLE_RATIO"
    --curriculum-six-prob-start "$CURRICULUM_SIX_START"
    --curriculum-six-prob-end "$CURRICULUM_SIX_END"
    --curriculum-six-prob-episodes "$CURRICULUM_SIX_EPISODES"
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS"
)

if [ "${#ENV2_CHANNEL_ARGS[@]}" -gt 0 ]; then
    DQN_CMD+=("${ENV2_CHANNEL_ARGS[@]}")
fi

if [ "${#EPS_LINEAR_ARGS[@]}" -gt 0 ]; then
    DQN_CMD+=("${EPS_LINEAR_ARGS[@]}")
fi

if [ -n "$STEPS_PER_EPISODE" ]; then
    DQN_CMD+=(--steps-per-episode "$STEPS_PER_EPISODE")
fi

if [ "$USE_AMP" = true ]; then
    DQN_CMD+=(--use-amp)
    echo -e "${GREEN}AMP enabled: 2-3x speedup on modern GPUs${NC}"
fi

if [ "$DISABLE_TENSORBOARD" = true ]; then
    DQN_CMD+=(--disable-tensorboard)
    echo -e "${GREEN}TensorBoard disabled${NC}"
fi

if [ "$SKIP_SUPERVISED" = false ] && [ -f "$SUPERVISED_MODEL" ]; then
    DQN_CMD+=(--policy-init "$SUPERVISED_MODEL")
    echo -e "${GREEN}Using supervised warm-start: $SUPERVISED_MODEL${NC}"
fi

if [ "$USE_SIMPLE_REWARDS" = true ]; then
    DQN_CMD+=(--simple-rewards)
    echo -e "${GREEN}Using simplified reward structure${NC}"
fi

echo "Running: ${DQN_CMD[*]}"
"${DQN_CMD[@]}"
echo -e "${GREEN}✓ DQN training completed${NC}"
echo ""

################################################################################
# Step 5: Hold-out Evaluation (Env2)
################################################################################
echo -e "${CYAN}Step 5/5: Hold-out evaluation (env2)${NC}"

HOLDOUT_CSV="$RUN_DIR/holdout_eval_${RUN_ID}.csv"

if [ ! -f "data/dqn_test.csv" ]; then
    echo -e "${YELLOW}No test split found. Skipping evaluation.${NC}"
else
    # Build evaluation command
    EVAL_CMD=(python rl/solver/evaluate_holdout.py
        --model-path "$DQN_MODEL"
        --test-csv data/dqn_test.csv
        --output-csv "$HOLDOUT_CSV"
        --min-size "$MIN_SIZE"
        --max-size "$MAX_SIZE"
        --max-colors "$MAX_COLORS"
        --env2-reward "$ENV2_REWARD"
        --env2-channels "${ENV2_CHANNELS[@]}"
        --epsilon 0.0
        --epsilon-start "$EPSILON_START"
        --epsilon-end "$EPSILON_END"
        --epsilon-schedule "$EPSILON_SCHEDULE"
        --move-penalty -0.05
        --distance-bonus 0.35
        --complete-bonus "$COMPLETE_BONUS"
        --complete-sustain-bonus "$COMPLETE_SUSTAIN_BONUS"
        --complete-revert-penalty "$COMPLETE_REVERT_PENALTY"
        --solve-bonus "$SOLVE_BONUS"
        --solve-efficiency-bonus "$SOLVE_EFFICIENCY_BONUS"
        --constraint-free-bonus "$CONSTRAINT_FREE_BONUS"
        --unsolved-penalty "$UNSOLVED_PENALTY"
        --unsolved-penalty-start "$UNSOLVED_PENALTY_START"
        --unsolved-penalty-warmup "$UNSOLVED_PENALTY_WARMUP"
        --reward-scale 1.0
        --reward-clamp 5.0
        --disconnect-penalty "$DISCONNECT_PENALTY"
        --degree-penalty "$DEGREE_PENALTY"
        --penalty-warmup "$PENALTY_WARMUP"
        --undo-penalty "$UNDO_PENALTY"
        --loop-penalty "$LOOP_PENALTY"
        --loop-window "$LOOP_WINDOW"
        --progress-bonus "$PROGRESS_BONUS"
        --rollout-mode "$HOLDOUT_ROLLOUT_MODE"
        --rollout-dir "$HOLDOUT_ROLLOUT_DIR"
        --rollout-max "$HOLDOUT_ROLLOUT_MAX"
        --gif-duration "$HOLDOUT_GIF_DURATION"
        --rollout-tag "$RUN_ID"
        --seed "$SEED"
    )

    if [ -n "$EPSILON_LINEAR_STEPS" ]; then
        EVAL_CMD+=(--epsilon-linear-steps "$EPSILON_LINEAR_STEPS")
    fi

    if [ "$USE_SIMPLE_REWARDS" = true ]; then
        EVAL_CMD+=(--simple-rewards)
    fi

    if [ "$HOLDOUT_GIF" = true ]; then
        EVAL_CMD+=(--gif)
    fi

    if [ -n "$STEPS_PER_EPISODE" ]; then
        EVAL_CMD+=(--steps-per-episode "$STEPS_PER_EPISODE")
    fi

    # Run evaluation
    "${EVAL_CMD[@]}" | tee "$RUN_DIR/test_results.txt"
fi

echo -e "${GREEN}✓ Pipeline complete${NC}"

################################################################################
# Summary
################################################################################
echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                     Env2 Pipeline Done                     ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "Run ID: $RUN_ID"
echo "Model: $DQN_MODEL"
echo "Logs:  $RUN_DIR"
echo "Holdout metrics: $HOLDOUT_CSV"
echo "Holdout rollouts (mode: $HOLDOUT_ROLLOUT_MODE): $HOLDOUT_ROLLOUT_DIR"
echo "MLflow: cd $RUN_DIR && mlflow ui --backend-store-uri file://./mlruns --port 5000"
echo "Test results: $RUN_DIR/test_results.txt"
echo ""
