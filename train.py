from pathlib import Path

import time
print(f"{time.time():.2f}: train.py script started")

# Apply compatibility patch for diffusers/transformers before any other imports
import fix_diffusers_deepspeed

print(f"{time.time():.2f}: Importing hydra and omegaconf...")
import hydra
from omegaconf import DictConfig
print(f"{time.time():.2f}: Finished importing hydra and omegaconf.")

from robobase.workspace import Workspace
print(f"{time.time():.2f}: Finished importing Workspace.")


@hydra.main(config_path="robobase/cfgs", config_name="robobase_config", version_base="1.3")
def main(cfg: DictConfig):
    # We don't need to set the wandb name here, because it is set in the config file
    # based on the experiment name and the current time.
    print(f"{time.time():.2f}: main(cfg) started")
    workspace = Workspace(cfg)
    workspace.train()


if __name__ == "__main__":
    print(f"{time.time():.2f}: Calling hydra.main...")
    main()
