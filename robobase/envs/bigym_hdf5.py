"""
Custom BigYM environment factory for loading demos from converted HDF5 files.

This factory extends BiGymEnvFactory to load demonstrations from pickle files
converted from HDF5 format instead of using DemoStore.
"""

import logging
import pickle
from pathlib import Path
from typing import List
import numpy as np
from omegaconf import DictConfig

# Import BiGym
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / 'bigym'))

from bigym.bigym_env import CONTROL_FREQUENCY_MAX
from demonstrations.demo import Demo, DemoStep
from demonstrations.utils import Metadata

from robobase.envs.bigym import BiGymEnvFactory


class BiGymHDF5EnvFactory(BiGymEnvFactory):
    """
    Custom BiGym environment factory for loading demos from converted HDF5.
    
    This factory loads demos from pickle files (converted from HDF5)
    instead of using BigYM's DemoStore.
    """
    
    def __init__(self):
        super().__init__()
        self._val_demos = None  # Store validation demos separately
    
    def _compute_action_stats(self, cfg: DictConfig, demos: List):
        """
        Compute action statistics from HDF5 demos.
        
        Both HDF5 demos and BiGym env now have 16 dimensions:
        - Base: 3 (X, Y, RZ)
        - Torso: 1
        - Left arm: 5
        - Left gripper: 1
        - Right arm: 5
        - Right gripper: 1
        
        No slicing needed - dimensions match perfectly after enabling
        torso in H1 config.
        """
        actions = []
        for demo in demos:
            for step in demo.timesteps:
                info = step.info
                if "demo_action" in info:
                    actions.append(info["demo_action"])
        actions = np.stack(actions)
        
        logging.info(f"HDF5 demo actions shape: {actions.shape}")
        
        # Compute stats directly
        action_mean = np.mean(actions, 0)
        action_std = np.std(actions, 0)
        action_max = np.max(actions, 0)
        action_min = np.min(actions, 0)
        
        logging.info(f"Computed action_stats shapes:")
        logging.info(f"  mean: {action_mean.shape}, std: {action_std.shape}")
        logging.info(f"  max: {action_max.shape}, min: {action_min.shape}")
        
        action_stats = {
            "mean": action_mean,
            "std": action_std,
            "max": action_max,
            "min": action_min,
        }
        return action_stats
    
    def post_collect_or_fetch_demos(self, cfg: DictConfig):
        """
        Process demos after loading.
        
        No action slicing needed - HDF5 16-dim matches BiGym 16-dim.
        Just rescale actions to normalized space.
        """
        from robobase.envs.bigym import rescale_demo_actions
        
        demo_list = [demo.timesteps for demo in self._raw_demos]
        
        logging.info("Actions already match BiGym action space (16 dims with torso)")
        
        # Rescale actions to normalized space
        demo_list = rescale_demo_actions(
            self._rescale_demo_action_helper, demo_list, cfg
        )
        self._demos = self._demo_to_steps(cfg, demo_list)
        
        # Also process validation demos if they exist
        if self._val_demos is not None:
            val_demo_list = [demo.timesteps for demo in self._val_demos]
            val_demo_list = rescale_demo_actions(
                self._rescale_demo_action_helper, val_demo_list, cfg
            )
            self._val_demos = self._demo_to_steps(cfg, val_demo_list)
            logging.info(f"Processed {len(self._val_demos)} validation demos (returns list of demos, not steps)")
    
    def get_val_demos(self):
        """Return validation demos."""
        return self._val_demos
    
    def load_val_demos_into_replay(self, cfg: DictConfig, buffer, is_demo_buffer):
        """Load validation demos into replay buffer."""
        import copy
        from robobase.envs.env import DemoEnv
        from robobase.utils import add_demo_to_replay_buffer
        
        if self._val_demos is None or len(self._val_demos) == 0:
            logging.warning("No validation demos to load")
            return
        
        logging.info(f"Loading {len(self._val_demos)} validation demos into buffer...")
        
        # Create DemoEnv wrapper - _val_demos should be in steps format already
        demo_env = self._wrap_env(
            DemoEnv(
                copy.deepcopy(self._val_demos), self._action_space, self._observation_space
            ),
            cfg,
            demo_env=True,
            train=False,
        )
        
        # Add each demo
        num_demos_added = 0
        for i in range(len(self._val_demos)):
            add_demo_to_replay_buffer(demo_env, buffer)
            num_demos_added += 1
        
        logging.info(f"Loaded {num_demos_added} validation demos with {len(buffer)} transitions")
    
    def _get_demo_fn(self, cfg: DictConfig, num_demos: int):
        """
        Load demos from converted pickle files with train/val split.
        
        Args:
            cfg: Configuration
            num_demos: Number of demos to load (-1 for all)
            
        Returns:
            List of Demo objects (training set only)
        """
        demo_path = Path(cfg.get('demo_path', './converted_demos/saucepan_full/'))
        
        logging.info(f"Loading demos from converted HDF5: {demo_path}")
        
        # Get all demo pickle files
        demo_files = sorted(demo_path.glob('*.pkl'), 
                          key=lambda x: int(x.stem.split('_')[1]))
        
        if not demo_files:
            raise ValueError(f"No demo files found in {demo_path}")
        
        total_demos = len(demo_files)
        if num_demos > 0 and num_demos < total_demos:
            demo_files = demo_files[:num_demos]
            total_demos = num_demos
        
        # Split into train/val (9:1 ratio)
        # 23 demos -> 21 train, 2 val
        # 20 demos -> 18 train, 2 val
        val_ratio = 0.1
        num_val = max(1, int(total_demos * val_ratio))  # At least 1 for validation
        num_train = total_demos - num_val
        
        train_files = demo_files[:num_train]
        val_files = demo_files[num_train:]
        
        logging.info(f"Dataset split: {num_train} train, {num_val} val (total: {total_demos})")
        logging.info(f"Train demos: {[f.stem for f in train_files]}")
        logging.info(f"Val demos: {[f.stem for f in val_files]}")
        
        # Create environment for metadata
        env = self._create_env(cfg)
        metadata = Metadata.from_env(env)
        
        # Load training demos
        train_demos = []
        for demo_file in train_files:
            # Load timesteps from pickle
            with open(demo_file, 'rb') as f:
                timesteps_data = pickle.load(f)
            
            # Convert to DemoStep objects
            demo_timesteps = []
            for i, step_data in enumerate(timesteps_data):
                if i == 0:
                    # First step: (observation, info)
                    observation, info = step_data
                    # Create a dummy first step with action from info
                    action = info['demo_action']
                    demo_step = DemoStep(
                        observation=observation,
                        reward=0.0,
                        termination=False,
                        truncation=False,
                        info=info,
                        action=action,
                    )
                    # Remove depth images (not used by robobase)
                    for key in list(demo_step.observation.keys()):
                        if key.startswith('depth_'):
                            del demo_step.observation[key]
                else:
                    # Subsequent steps: (observation, reward, term, trunc, info)
                    observation, reward, termination, truncation, info = step_data
                    action = info['demo_action']
                    demo_step = DemoStep(
                        observation=observation,
                        reward=reward,
                        termination=termination,
                        truncation=truncation,
                        info=info,
                        action=action,
                    )
                    # Remove depth images (not used by robobase)
                    for key in list(demo_step.observation.keys()):
                        if key.startswith('depth_'):
                            del demo_step.observation[key]
                
                demo_timesteps.append(demo_step)
            
            # Create Demo object
            demo = Demo(
                metadata=metadata,
                timesteps=demo_timesteps,
            )
            train_demos.append(demo)
            
            logging.info(f"Loaded {demo_file.name}: {len(demo_timesteps)} steps")
        
        # Load validation demos
        val_demos = []
        for demo_file in val_files:
            # Load timesteps from pickle
            with open(demo_file, 'rb') as f:
                timesteps_data = pickle.load(f)
            
            # Convert to DemoStep objects
            demo_timesteps = []
            for i, step_data in enumerate(timesteps_data):
                if i == 0:
                    # First step: (observation, info)
                    observation, info = step_data
                    action = info['demo_action']
                    demo_step = DemoStep(
                        observation=observation,
                        reward=0.0,
                        termination=False,
                        truncation=False,
                        info=info,
                        action=action,
                    )
                    # Remove depth images
                    for key in list(demo_step.observation.keys()):
                        if key.startswith('depth_'):
                            del demo_step.observation[key]
                else:
                    # Subsequent steps
                    observation, reward, termination, truncation, info = step_data
                    action = info['demo_action']
                    demo_step = DemoStep(
                        observation=observation,
                        reward=reward,
                        termination=termination,
                        truncation=truncation,
                        info=info,
                        action=action,
                    )
                    # Remove depth images
                    for key in list(demo_step.observation.keys()):
                        if key.startswith('depth_'):
                            del demo_step.observation[key]
                
                demo_timesteps.append(demo_step)
            
            # Create Demo object
            demo = Demo(
                metadata=metadata,
                timesteps=demo_timesteps,
            )
            val_demos.append(demo)
            
            logging.info(f"Loaded validation {demo_file.name}: {len(demo_timesteps)} steps")
        
        env.close()
        
        # Store validation demos for later use
        self._val_demos = val_demos
        
        logging.info(f"Finished loading {len(train_demos)} train demos and {len(val_demos)} val demos.")
        
        return train_demos


def create_hdf5_env_factory():
    """Factory function to create BiGymHDF5EnvFactory."""
    return BiGymHDF5EnvFactory()
