#!/bin/bash

# Diffusion Policy Training Script for BigYM HDF5 Demonstrations
# Uses RGB images (224x224) and 16D proprioception from demos.hdf5
# Same data format as ACT training
#
# Data Format (BigYM Native 16D):
#   - RGB: rgb_head (3, 224, 224) - head camera
#   - Proprioception (16D): [floating_base(4), left_arm(5), right_arm(5), grippers(2)]
#   - Actions (16D): same structure as proprioception
#
# Model Architecture:
#   - ResNet18 encoder for RGB images (pretrained on ImageNet)
#   - ConditionalUnet1D with proper skip connections
#   - DDIM scheduler with squaredcos_cap_v2 beta schedule
#
# Usage:
#   ./train_diffusion.sh                              # Default training
#   ./train_diffusion.sh --batch-size 128            # Custom batch size
#   ./train_diffusion.sh --epochs 500 --no-wandb     # Local training without W&B

echo "=========================================="
echo "Diffusion Policy Training - BigYM"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bigym_test_1

cd /home/hyunjin/bigym_ws/robobase

# Data path - Modify this for different tasks
DATA_PATH=/home/hyunjin/bigym_ws/data/demonstrations/saucepan_to_hob/demos.hdf5
ACTION_STATS_PATH=/home/hyunjin/bigym_ws/data/demonstrations/saucepan_to_hob/action_stats.json

# Training configuration
# --preload-all: Loads all data into RAM (~5GB) for faster training
python -m robobase.method.diffusion_lightning \
    --hdf5-path $DATA_PATH \
    --action-stats-path $ACTION_STATS_PATH \
    --batch-size 256 \
    --lr 1e-4 \
    --epochs 2000 \
    --action-sequence 16 \
    --image-size 224 \
    --frame-stack 1 \
    --num-workers 16 \
    --preload-all \
    --eval-every 100 \
    --eval-episodes 1 \
    --eval-max-steps 1000 \
    --log-eval-video \
    --num-train-timesteps 100 \
    --num-inference-steps 16 \
    --gradient-clip 1.0 \
    --run-name diffusion_$(date +%Y%m%d_%H%M%S) \
    --wandb \
    --wandb-project diffusion_bigym \
    "$@"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
