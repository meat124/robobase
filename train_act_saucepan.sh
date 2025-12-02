#!/bin/bash

# ACT Policy Training Script for BigYM HDF5 Demonstrations
# Uses RGB images (224x224) and 16D proprioception from demos.hdf5
#
# Data Format (BigYM Native 16D):
#   - RGB: rgb_head (3, 224, 224) - head camera
#   - Proprioception (16D): [floating_base(4), left_arm(5), right_arm(5), grippers(2)]
#   - Actions (16D): same structure as proprioception
#
# Usage:
#   ./train_act_saucepan.sh                           # Default SaucepanToHob training
#   ./train_act_saucepan.sh --batch-size 128         # Custom batch size
#   ./train_act_saucepan.sh --epochs 500 --no-wandb  # Local training without W&B

echo "=========================================="
echo "ACT Policy Training - BigYM"
echo "=========================================="

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bigym_test_1

cd /home/hyunjin/bigym_ws/robobase

# Data path - Modify this for different tasks
DATA_PATH=/home/hyunjin/bigym_ws/data/demonstrations/saucepan_to_hob/demos.hdf5

# Training configuration
# --preload-all: Loads all data into RAM (~5GB) for faster training
python train_act_hdf5.py \
    --hdf5-path $DATA_PATH \
    --batch-size 256 \
    --lr 1e-4 \
    --epochs 1000 \
    --action-sequence 16 \
    --image-size 224 \
    --eval-every 50 \
    --eval-episodes 1 \
    --frame-stack 1 \
    --num-workers 8 \
    --preload-all \
    --run-name act_saucepan_$(date +%Y%m%d_%H%M%S) \
    --wandb \
    --wandb-project act_bigym \
    "$@"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
