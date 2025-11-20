#!/bin/bash

# Run training with PCD dataset
python -m robobase.method.brs_lightning \
    --config robobase/cfgs/brs_config.yaml \
    --use-pcd \
    --bs 256 \
    --vbs 256 \
    --wandb-name brs_policy_train_test_4 \
    "$@"
