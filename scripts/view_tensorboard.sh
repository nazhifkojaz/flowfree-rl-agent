#!/bin/bash
################################################################################
# View Training Progress in TensorBoard
#
# This script starts TensorBoard to monitor training progress
################################################################################

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

# Default to full pipeline logs
LOG_DIR="${1:-logs/dqn_full_pipeline_env2/tensorboard}"
PORT="${2:-6006}"

# Check if TensorBoard directory exists
if [ ! -d "$LOG_DIR" ]; then
    echo "Error: TensorBoard directory $LOG_DIR does not exist"
    echo ""
    echo "Searching for TensorBoard directories..."
    find logs -name "tensorboard" -type d 2>/dev/null | head -10
    echo ""
    echo "If no directories found, training may not have started yet."
    echo "TensorBoard logging is automatically enabled when training starts."
    exit 1
fi

# Check if there are event files
EVENT_COUNT=$(find "$LOG_DIR" -name "events.out.tfevents.*" 2>/dev/null | wc -l)
if [ "$EVENT_COUNT" -eq 0 ]; then
    echo "Warning: No TensorBoard event files found in $LOG_DIR"
    echo "Training may not have logged any data yet."
    echo ""
    echo "Press Ctrl+C to cancel, or wait for TensorBoard to start..."
    sleep 2
fi

echo "Starting TensorBoard..."
echo "Log directory: $LOG_DIR"
echo "Port: $PORT"
echo ""
echo "Open browser to: http://localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Start TensorBoard
tensorboard --logdir "$LOG_DIR" --port "$PORT"
