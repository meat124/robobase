#!/bin/bash

# Diffusion Policy Training Script for BiGym HDF5 Dataset
# Trains on full dataset: 22 demos with 84x84 RGB images (no depth)

echo "=========================================="
echo "BiGym Diffusion Policy Training"
echo "=========================================="
echo ""

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bigym_test_1

# Change to robobase directory
cd /home/hyunjin/bigym_ws/robobase

echo "Dataset: 22 demonstrations (84x84 RGB, no depth)"
echo "Training steps: 100,000"
echo "Batch size: 128"
echo ""
echo "Starting training..."
echo "=========================================="
echo ""

# Run training
python train.py launch=dp_pixel_bigym_hdf5

echo ""
echo "=========================================="
echo "Training completed or terminated"
echo "=========================================="
