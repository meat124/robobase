#!/bin/bash

# ============================================================================
# BRS Policy Training Script
# ============================================================================
#
# Uses BigYM native 16D action format:
#   - BigYM actions: [floating_base(4), arms(12)] = 16D
#   - BRS structure: [mobile_base(3), torso(1), arms(12)] = 16D
#
# Supports multiple tasks: saucepan, flipcup, etc.
#
# Usage:
#   ./train_brs.sh                        # Default: flipcup
#   ./train_brs.sh --task flipcup         # Train on FlipCup
#   ./train_brs.sh --task saucepan        # Train on SaucepanToHob
#   ./train_brs.sh --debug                # Debug mode (small batch, no wandb)
#   ./train_brs.sh --resume <ckpt>        # Resume from checkpoint
#   ./train_brs.sh --bs 128               # Custom batch size
#
# ============================================================================

set -e

# ============================================
# Default Configuration
# ============================================
TASK="flipcup"
DATA_ROOT="/home/hyunjin/bigym_ws/data/demonstrations"

# Training hyperparameters
BATCH_SIZE=384
VAL_BATCH_SIZE=384
NUM_WORKERS=16
PREFETCH_FACTOR=4

# ============================================
# Parse arguments
# ============================================
DEBUG_MODE=false
RESUME_CKPT=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --task|-t)
            TASK="$2"
            shift 2
            ;;
        --debug)
            DEBUG_MODE=true
            shift
            ;;
        --resume)
            RESUME_CKPT="$2"
            shift 2
            ;;
        --bs)
            BATCH_SIZE="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# ============================================
# Task-specific configuration
# ============================================
case $TASK in
    saucepan|saucepan_to_hob)
        TASK_NAME="SaucepanToHob"
        DATA_DIR="${DATA_ROOT}/saucepan_to_hob"
        ;;
    flipcup|flip_cup)
        TASK_NAME="FlipCup"
        DATA_DIR="${DATA_ROOT}/flipcup"
        ;;
    *)
        # Allow custom data directory
        TASK_NAME="$TASK"
        DATA_DIR="${DATA_ROOT}/${TASK}"
        ;;
esac

HDF5_PATH="${DATA_DIR}/demos.hdf5"
PCD_ROOT="${DATA_DIR}/pcd"
CONFIG_PATH="robobase/cfgs/brs_config.yaml"

# ============================================
# Validate data
# ============================================
echo "=========================================="
echo "BRS Policy Training - ${TASK_NAME}"
echo "=========================================="
echo ""

if [ ! -f "$HDF5_PATH" ]; then
    echo "ERROR: HDF5 file not found: $HDF5_PATH"
    echo ""
    echo "Available tasks in ${DATA_ROOT}:"
    ls -d ${DATA_ROOT}/*/ 2>/dev/null | xargs -n1 basename || echo "  (none)"
    echo ""
    echo "Please run convert_demo_to_hdf5.py first or specify a valid task."
    exit 1
fi

if [ ! -d "$PCD_ROOT" ]; then
    echo "ERROR: PCD directory not found: $PCD_ROOT"
    exit 1
fi

echo "✓ Task:      $TASK_NAME"
echo "✓ Data dir:  $DATA_DIR"
echo "✓ HDF5:      $HDF5_PATH"
echo "✓ PCD root:  $PCD_ROOT"
echo "✓ Format:    BigYM Native (16D actions)"
echo ""

# ============================================
# Activate conda environment
# ============================================
source ~/miniconda3/etc/profile.d/conda.sh
conda activate bigym_test_1

cd /home/hyunjin/bigym_ws/robobase

# ============================================
# Debug mode settings
# ============================================
if [ "$DEBUG_MODE" = true ]; then
    echo ">>> DEBUG MODE ENABLED <<<"
    BATCH_SIZE=8
    VAL_BATCH_SIZE=8
    NUM_WORKERS=2
    WANDB_ARGS="--no-wandb"
    PRELOAD_ARGS=""
else
    WANDB_ARGS="--wandb-name brs_${TASK}_$(date +%Y%m%d_%H%M%S)"
    PRELOAD_ARGS="--preload-hdf5 --preload-pcd"
fi

# ============================================
# Resume from checkpoint
# ============================================
RESUME_ARGS=""
if [ -n "$RESUME_CKPT" ]; then
    if [ ! -f "$RESUME_CKPT" ]; then
        echo "ERROR: Checkpoint not found: $RESUME_CKPT"
        exit 1
    fi
    echo "Resuming from: $RESUME_CKPT"
    RESUME_ARGS="--resume $RESUME_CKPT"
fi

# ============================================
# Print configuration
# ============================================
echo "----------------------------------------"
echo "Training Configuration:"
echo "  Batch size:     $BATCH_SIZE"
echo "  Num workers:    $NUM_WORKERS"
echo "  Config:         $CONFIG_PATH"
echo "----------------------------------------"
echo ""

# ============================================
# Run training
# ============================================
python -m robobase.method.brs_lightning \
    --config "$CONFIG_PATH" \
    --hdf5-path "$HDF5_PATH" \
    --pcd-root "$PCD_ROOT" \
    --use-pcd \
    --bs $BATCH_SIZE \
    --vbs $VAL_BATCH_SIZE \
    --dataloader-num-workers $NUM_WORKERS \
    --dataloader-prefetch-factor $PREFETCH_FACTOR \
    --dataloader-persistent-workers \
    --use_torch_compile \
    $PRELOAD_ARGS \
    $WANDB_ARGS \
    $RESUME_ARGS \
    $EXTRA_ARGS

echo ""
echo "=========================================="
echo "Training completed!"
echo "=========================================="
