#!/bin/bash

# BRS Policy Training Script (RGB version) for SaucepanToHob
# Uses RGB images (ResNet18 encoder) instead of Point Cloud (PointNet)
#
# Dataset: saucepan_to_hob
#   - 31 demos, ~27k timesteps
#   - RGB: rgb_head (T+1, 3, 224, 224) uint8 in HDF5
#   - Proprioception (16D):
#       - mobile_base (3D): proprioception_floating_base [x, y, rz] absolute position
#       - torso (1D): proprioception_floating_base[2] (z position)
#       - left_arm (5D): qpos[0, 1, 2, 3, 12]
#       - left_gripper (1D): proprioception_grippers[0]
#       - right_arm (5D): qpos[13, 14, 15, 16, 25]
#       - right_gripper (1D): proprioception_grippers[1]
#   - Actions (16D): [mobile_base(3), torso(1), arms(12)] as DELTA
#
# Model: WBVIMAPolicyRGB
#   - ResNet18 encoder (pretrained ImageNet) ~11M params
#   - Same Transformer + Diffusion Head as PCD version
#   - Total: ~48M params (vs ~7.5M for PCD version)
#
# Note: RGB images are larger than PCD, so batch size is reduced

echo "=========================================="
echo "BRS Policy Training (RGB) - SaucepanToHob"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bigym_test_1

cd /home/hyunjin/bigym_ws/robobase

# Data paths
DATA_DIR=/home/hyunjin/bigym_ws/data/demonstrations/saucepan_to_hob
HDF5_PATH=${DATA_DIR}/demos.hdf5

# Run training with RGB dataset
# Note: Smaller batch size (64) due to larger RGB data
# --preload-data: Load all RGB into RAM (~4-5GB for 30 demos) - optional
python -m robobase.method.brs_lightning \
    --config robobase/cfgs/brs_config_rgb.yaml \
    --hdf5-path $HDF5_PATH \
    --use-rgb \
    --bs 128 \
    --vbs 128 \
    --preload-data \
    --dataloader-num-workers 8 \
    --no-wandb \
    "$@"

    # --wandb-name brs_rgb_saucepan_$(date +%Y%m%d_%H%M%S) \


echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
