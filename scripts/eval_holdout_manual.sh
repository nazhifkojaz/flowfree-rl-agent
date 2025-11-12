#!/bin/bash
set -euo pipefail

MODEL_PATH="${1:-models/latest.pt}"
ROLLOUT_MODE="${2:-both}"
ROLLOUT_DIR="${3:-logs/manual_holdout_rollouts}"
ROLLOUT_MAX="${4:-0}"
ROLLOUT_GIF="${5:-false}"
ROLLOUT_GIF_DURATION="${6:-140}"
LOOP_PENALTY="${7:--0.5}"
LOOP_WINDOW="${8:-6}"
PROGRESS_BONUS="${9:-0.02}"

RUN_ID=$(basename "$MODEL_PATH" .pt)
OUT_DIR="logs/manual_holdout"
OUT_CSV="$OUT_DIR/holdout_${RUN_ID}.csv"
mkdir -p "$OUT_DIR"

# Default configuration matching pipeline
MIN_SIZE=5
MAX_SIZE=5
MAX_COLORS=10
ENV2_CHANNELS=("occupancy" "endpoints" "heads" "free" "congestion" "distance")
DISTANCE_BONUS=0.35
MOVE_PENALTY=-0.05
UNSOLVED_PENALTY=-2.0
UNSOLVED_START=0.0
UNSOLVED_WARMUP=500
DISCONNECT_PENALTY=-0.20
DEGREE_PENALTY=-0.25
COMPLETE_BONUS=1.8
COMPLETE_SUSTAIN=0.1
COMPLETE_REVERT=2.0
SOLVE_BONUS=35.0
CONSTRAINT_FREE_BONUS=5.0
PENALTY_WARMUP=400
UNDO_PENALTY=-0.10
STEP_LIMIT=$((MAX_SIZE * MAX_SIZE * 2))

# Build evaluation command
EVAL_CMD=(python rl/solver/evaluate_holdout.py
    --model-path "$MODEL_PATH"
    --test-csv data/dqn_test.csv
    --output-csv "$OUT_CSV"
    --min-size "$MIN_SIZE"
    --max-size "$MAX_SIZE"
    --max-colors "$MAX_COLORS"
    --env2-reward potential
    --env2-channels "${ENV2_CHANNELS[@]}"
    --epsilon 0.0
    --epsilon-start 1.0
    --epsilon-end 0.10
    --epsilon-schedule linear
    --epsilon-linear-steps 4000
    --move-penalty "$MOVE_PENALTY"
    --distance-bonus "$DISTANCE_BONUS"
    --complete-bonus "$COMPLETE_BONUS"
    --complete-sustain-bonus "$COMPLETE_SUSTAIN"
    --complete-revert-penalty "$COMPLETE_REVERT"
    --solve-bonus "$SOLVE_BONUS"
    --constraint-free-bonus "$CONSTRAINT_FREE_BONUS"
    --unsolved-penalty "$UNSOLVED_PENALTY"
    --unsolved-penalty-start "$UNSOLVED_START"
    --unsolved-penalty-warmup "$UNSOLVED_WARMUP"
    --reward-scale 1.0
    --reward-clamp 5.0
    --disconnect-penalty "$DISCONNECT_PENALTY"
    --degree-penalty "$DEGREE_PENALTY"
    --penalty-warmup "$PENALTY_WARMUP"
    --undo-penalty "$UNDO_PENALTY"
    --loop-penalty "$LOOP_PENALTY"
    --loop-window "$LOOP_WINDOW"
    --progress-bonus "$PROGRESS_BONUS"
    --steps-per-episode "$STEP_LIMIT"
    --rollout-mode "$ROLLOUT_MODE"
    --rollout-dir "$ROLLOUT_DIR/$RUN_ID"
    --rollout-max "$ROLLOUT_MAX"
    --gif-duration "$ROLLOUT_GIF_DURATION"
    --rollout-tag "holdout_manual"
    --seed 42
)

if [ "$ROLLOUT_GIF" = "true" ]; then
    EVAL_CMD+=(--gif)
fi

# Run evaluation
"${EVAL_CMD[@]}"
