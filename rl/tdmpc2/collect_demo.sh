#!/bin/bash

# Data collection script for TD-MPC2 trained checkpoint
# Usage: bash collect_demo.sh

# Configuration
CHECKPOINT="/root/grasp-cube-sample/rl/tdmpc2/logs/StackCubeSO101-v1/1/default/models/final.pt"
NUM_EPISODES=100
OUTPUT_DIR="./demo_data"
SAVE_VIDEO=true
SAVE_ONLY_SUCCESS=true

echo "=================================================="
echo "TD-MPC2 Data Collection Script"
echo "=================================================="
echo "Checkpoint: $CHECKPOINT"
echo "Episodes to collect: $NUM_EPISODES"
echo "Output directory: $OUTPUT_DIR"
echo "Save videos: $SAVE_VIDEO"
echo "Save only successful episodes: $SAVE_ONLY_SUCCESS"
echo "=================================================="
echo ""

# Run data collection
# Note: Use obs=rgb + include_state=true to match the checkpoint training mode
python collect_data.py \
    checkpoint=$CHECKPOINT \
    obs=rgb \
    include_state=true \
    num_envs=1 \
    num_eval_envs=1 \
    +num_episodes=$NUM_EPISODES \
    +output_dir=$OUTPUT_DIR \
    +save_video=$SAVE_VIDEO \
    +save_only_success=$SAVE_ONLY_SUCCESS \
    model_size=5

echo ""
echo "Data collection complete!"
echo "Check the output directory: $OUTPUT_DIR"

