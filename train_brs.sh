#!/bin/bash

# BRS Policy Training Script for SaucepanToHob (Full Observations)
# Uses Point Cloud (PCD) and 16D proprioception from demos.hdf5
#
# Dataset: saucepan_to_hob
#   - 31 demos, ~27k timesteps
#   - PCD: pcd/demo_XXX_pcd.npy (4096, 3) per frame
#   - Proprioception (16D):
#       - mobile_base (3D): proprioception_floating_base [x, y, rz] absolute position
#       - torso (1D): proprioception_floating_base[2] (z position)
#       - left_arm (5D): qpos[0, 1, 2, 3, 12]
#       - left_gripper (1D): proprioception_grippers[0]
#       - right_arm (5D): qpos[13, 14, 15, 16, 25]
#       - right_gripper (1D): proprioception_grippers[1]
#   - Actions (16D): [mobile_base(3), torso(1), arms(12)] as DELTA (NOT velocity)
#
# Note: Uses PCDBRSDataset with new .npy PCD format
#
# Preload Options:
#   --preload-hdf5  : Load all HDF5 data into RAM (~100MB for 30 demos) - RECOMMENDED
#   --preload-pcd   : Load all PCD data into RAM (~3GB for 30 demos) - if RAM > 8GB

echo "=========================================="
echo "BRS Policy Training - SaucepanToHob"
echo "=========================================="

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bigym_test_1

cd /home/hyunjin/bigym_ws/robobase

# Data paths (new full observations dataset)
DATA_DIR=/home/hyunjin/bigym_ws/data/demonstrations/saucepan_to_hob
HDF5_PATH=${DATA_DIR}/demos.hdf5
PCD_ROOT=${DATA_DIR}/pcd

# Run training with PCD dataset
# Optimized data loading: num_workers=8, prefetch_factor=4, persistent_workers
# Preload options: --preload-hdf5 (recommended) or --preload-pcd (if RAM > 8GB)
python -m robobase.method.brs_lightning \
    --config robobase/cfgs/brs_config.yaml \
    --hdf5-path $HDF5_PATH \
    --pcd-root $PCD_ROOT \
    --use-pcd \
    --bs 256 \
    --vbs 256 \
    --dataloader-num-workers 8 \
    --dataloader-prefetch-factor 4 \
    --dataloader-persistent-workers \
    --preload-hdf5 \
    --preload-pcd \
    --wandb-name brs_saucepan_$(date +%Y%m%d_%H%M%S) \
    "$@"

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="