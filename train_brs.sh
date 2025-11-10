#!/bin/bash

# Run training with PCD dataset
python -m robobase.method.brs_lightning \
    --config robobase/cfgs/brs_config.yaml \
    --use-pcd \
    --bs 512 \
    --vbs 512 \
    "$@"
