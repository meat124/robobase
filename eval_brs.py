#!/usr/bin/env python3
"""
Evaluation script for BRS policy on BigYM environment.
Supports both PCD and RGB observation modes.

Usage:
    # Auto-find latest checkpoint
    python eval_brs.py --exp-dir exp_local/brs_saucepan --num-episodes 10
    
    # Specify checkpoint
    python eval_brs.py --checkpoint path/to/model.ckpt --num-episodes 10
    
    # RGB mode
    python eval_brs.py --exp-dir exp_local/brs_rgb_saucepan --use-rgb --num-episodes 10
    
    # With video saving
    python eval_brs.py --exp-dir exp_local/brs_saucepan --save-video --video-dir eval_videos

Action Space Notes (from README_HDF5_MAPPING.md):
    - Demo/BRS Policy outputs 16D actions:
        [0:4]   floating_base (pelvis_x, pelvis_y, pelvis_z, pelvis_rz) - DELTA mode
        [4:14]  arms (10 joints) - ABSOLUTE mode
        [14:16] grippers (2) - COMMAND mode
    
    - Environment with torso=False: 16D (matches demo)
    - Environment with torso=True: 17D (insert torso=0 at index 4)
    
    - CRITICAL: Policy outputs 50Hz actions, env runs at 500Hz (decimation=10)
        - Floating base deltas must be DIVIDED by 10
        - Execute same action 10 times (or use action repeat)
"""

import argparse
import os
import glob
import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from tqdm import tqdm

# BigYM imports
import bigym
from bigym.action_modes import JointPositionActionMode, PelvisDof
from bigym.envs.move_plates import MovePlate, MoveTwoPlates
from bigym.envs.pick_and_place import (
    PickBox, SaucepanToHob, PutCups, TakeCups, StoreBox, 
    StoreKitchenware, ToastSandwich, FlipSandwich, RemoveSandwich
)
from bigym.envs.reach_target import ReachTarget, ReachTargetDual, ReachTargetSingle
from bigym.envs.manipulation import FlipCup, FlipCutlery, StackBlocks
from bigym.envs.groceries import GroceriesStoreLower, GroceriesStoreUpper
from bigym.robots.configs.h1 import H1FineManipulation
from bigym.utils.observation_config import ObservationConfig, CameraConfig

# Optional video recording
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

# Decimation factor (500Hz env / 50Hz policy)
ENV_DECIMATION = 10


# Environment mapping (only include environments that exist in BigYM)
ENV_REGISTRY = {
    # Pick and place
    "pick_box": PickBox,
    "saucepan_to_hob": SaucepanToHob,
    "put_cups": PutCups,
    "take_cups": TakeCups,
    "store_box": StoreBox,
    "store_kitchenware": StoreKitchenware,
    "toast_sandwich": ToastSandwich,
    "flip_sandwich": FlipSandwich,
    "remove_sandwich": RemoveSandwich,
    # Manipulation
    "flip_cup": FlipCup,
    "flip_cutlery": FlipCutlery,
    "stack_blocks": StackBlocks,
    # Groceries
    "groceries_store_lower": GroceriesStoreLower,
    "groceries_store_upper": GroceriesStoreUpper,
    # Reach
    "reach_target": ReachTarget,
    "reach_target_dual": ReachTargetDual,
    "reach_target_single": ReachTargetSingle,
    # Move plates
    "move_plate": MovePlate,
    "move_two_plates": MoveTwoPlates,
}


def find_latest_checkpoint(exp_dir: str) -> Optional[str]:
    """Find the latest checkpoint in experiment directory."""
    exp_path = Path(exp_dir)
    
    # Check common checkpoint locations
    patterns = [
        str(exp_path / "checkpoints" / "*.ckpt"),
        str(exp_path / "lightning_logs" / "*" / "checkpoints" / "*.ckpt"),
        str(exp_path / "**" / "*.ckpt"),
    ]
    
    checkpoints = []
    for pattern in patterns:
        checkpoints.extend(glob.glob(pattern, recursive=True))
    
    if not checkpoints:
        return None
    
    # Sort by modification time, return most recent
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    return checkpoints[0]


def load_checkpoint_and_config(checkpoint_path: str, device: str = "cuda") -> Tuple[Any, Dict]:
    """Load checkpoint and extract config."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Try to get config from checkpoint
    config = checkpoint.get("hyper_parameters", {}).get("config", None)
    if config is None:
        # Try alternative locations
        config = checkpoint.get("config", None)
    
    return checkpoint, config


def create_policy(
    checkpoint_path: str,
    use_rgb: bool = False,
    device: str = "cuda",
    config_path: Optional[str] = None,
) -> Tuple[Any, Dict, Optional[Dict]]:
    """
    Create and load policy from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        use_rgb: Whether to use RGB policy
        device: Device to use
        config_path: Path to config file. If None, will try to find it automatically.
    
    Returns:
        Tuple of (model, config, action_stats)
    """
    import yaml
    import json
    from robobase.method.brs_lightning import (
        SafeDiffusionModule, 
        create_module_from_config,
    )
    
    # Find config file
    if config_path is None:
        # Try to find config in experiment directory
        exp_dir = Path(checkpoint_path).parent.parent
        possible_configs = [
            exp_dir / "config.yaml",
            exp_dir / "brs_config.yaml",
            Path("robobase/cfgs/brs_config.yaml"),
            Path("robobase/cfgs/brs_config_rgb.yaml") if use_rgb else Path("robobase/cfgs/brs_config.yaml"),
        ]
        
        for cfg_path in possible_configs:
            if cfg_path.exists():
                config_path = str(cfg_path)
                print(f"Found config: {config_path}")
                break
        
        if config_path is None:
            raise ValueError(f"Could not find config file. Tried: {possible_configs}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Create module from config
    module = create_module_from_config(config, use_rgb=use_rgb)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load state dict
    module.load_state_dict(checkpoint['state_dict'])
    
    module.eval()
    module.to(device)
    
    print(f"Model loaded successfully from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Global step: {checkpoint.get('global_step', 'N/A')}")
    
    # Load all stats for normalization/denormalization
    stats = {
        'action_stats': None,
        'prop_stats': None,
        'pcd_stats': None,
    }
    exp_dir = Path(checkpoint_path).parent.parent
    
    # Load action stats
    if config.get('normalize', False):
        for path_key in ['action_stats_path']:
            path = config.get(path_key)
            local_path = exp_dir / "action_stats.json"
            if local_path.exists():
                with open(local_path, 'r') as f:
                    stats['action_stats'] = json.load(f)
                print(f"Loaded action stats from: {local_path}")
                break
            elif path and os.path.exists(path):
                with open(path, 'r') as f:
                    stats['action_stats'] = json.load(f)
                print(f"Loaded action stats from: {path}")
                break
    
    # Load prop stats
    if config.get('normalize', False):
        local_path = exp_dir / "prop_stats.json"
        if local_path.exists():
            with open(local_path, 'r') as f:
                stats['prop_stats'] = json.load(f)
            print(f"Loaded prop stats from: {local_path}")
        elif config.get('prop_stats_path') and os.path.exists(config['prop_stats_path']):
            with open(config['prop_stats_path'], 'r') as f:
                stats['prop_stats'] = json.load(f)
            print(f"Loaded prop stats from: {config['prop_stats_path']}")
    
    # Load pcd stats
    if config.get('normalize_pcd', False):
        local_path = exp_dir / "pcd_stats.json"
        if local_path.exists():
            with open(local_path, 'r') as f:
                stats['pcd_stats'] = json.load(f)
            print(f"Loaded pcd stats from: {local_path}")
        elif config.get('pcd_stats_path') and os.path.exists(config['pcd_stats_path']):
            with open(config['pcd_stats_path'], 'r') as f:
                stats['pcd_stats'] = json.load(f)
            print(f"Loaded pcd stats from: {config['pcd_stats_path']}")
    
    return module, config, stats


def denormalize_action(
    action_normalized: np.ndarray,
    action_stats: Dict[str, Any],
) -> np.ndarray:
    """
    Denormalize action from [-1, 1] to original range.
    
    Uses min-max normalization (matches training):
    normalized = 2.0 * (value - min) / (max - min) - 1.0
    
    Inverse: value = (normalized + 1) / 2 * (max - min) + min
    """
    full_stats = action_stats.get("full", action_stats)
    
    min_val = np.array(full_stats["min"], dtype=np.float32)
    max_val = np.array(full_stats["max"], dtype=np.float32)
    
    # Denormalize: value = (normalized + 1) / 2 * (max - min) + min
    range_val = max_val - min_val
    range_val = np.where(range_val < 1e-6, 1e-6, range_val)  # Avoid division by zero
    action = (action_normalized + 1.0) / 2.0 * range_val + min_val
    
    return action


class ObservationHistoryManager:
    """
    Manages observation history for BRS policy.
    
    BRS policy expects observations with time dimension (B, T, ...) where T = num_latest_obs.
    This class maintains a rolling buffer of observations.
    """
    
    def __init__(
        self,
        num_latest_obs: int = 2,
        n_points: int = 4096,
        use_rgb: bool = False,
        normalize_prop: bool = True,
        normalize_pcd: bool = True,
        prop_stats: Optional[Dict] = None,
        pcd_stats: Optional[Dict] = None,
        device: str = "cuda",
    ):
        self.num_latest_obs = num_latest_obs
        self.n_points = n_points
        self.use_rgb = use_rgb
        self.normalize_prop = normalize_prop
        self.normalize_pcd = normalize_pcd
        self.prop_stats = prop_stats
        self.pcd_stats = pcd_stats
        self.device = device
        
        # History buffers (store numpy arrays)
        self.history = {
            'base_velocity': [],
            'torso': [],
            'left_arm': [],
            'left_gripper': [],
            'right_arm': [],
            'right_gripper': [],
        }
        if use_rgb:
            self.history['rgb'] = []
        else:
            self.history['pcd_xyz'] = []
            self.history['pcd_rgb'] = []
        
        # Previous floating base for velocity calculation
        self.prev_floating_base = None
        
        # Load normalization stats
        if self.prop_stats is not None:
            full_prop = self.prop_stats.get("full", self.prop_stats)
            self.prop_min = np.array(full_prop.get("min", [0]*16), dtype=np.float32)
            self.prop_max = np.array(full_prop.get("max", [1]*16), dtype=np.float32)
        else:
            self.prop_min = None
            self.prop_max = None
        
        if self.pcd_stats is not None:
            xyz_stats = self.pcd_stats.get("xyz", self.pcd_stats)
            self.pcd_xyz_min = np.array(xyz_stats.get("min", [-1, -1, 0]), dtype=np.float32)
            self.pcd_xyz_max = np.array(xyz_stats.get("max", [1, 1, 2]), dtype=np.float32)
        else:
            self.pcd_xyz_min = None
            self.pcd_xyz_max = None
    
    def reset(self):
        """Reset observation history."""
        for key in self.history:
            self.history[key] = []
        self.prev_floating_base = None
    
    def _normalize_to_range(self, value: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
        """Normalize value to [-1, 1] range."""
        range_val = max_val - min_val
        range_val = np.where(np.abs(range_val) < 1e-6, 1e-6, range_val)
        normalized = 2.0 * (value - min_val) / range_val - 1.0
        return np.clip(normalized, -1.0, 1.0)
    
    def _extract_features(self, obs: Dict[str, Any]) -> Dict[str, np.ndarray]:
        """Extract features from BigYM observation."""
        proprioception = obs.get("proprioception", np.zeros(60))
        proprioception_grippers = obs.get("proprioception_grippers", np.zeros(2))
        proprioception_floating_base = obs.get("proprioception_floating_base", np.zeros(4))
        
        qpos = proprioception[:30]
        
        QPOS_LEFT_ARM = [0, 1, 2, 3, 12]
        QPOS_RIGHT_ARM = [13, 14, 15, 16, 25]
        
        left_arm = qpos[QPOS_LEFT_ARM].astype(np.float32)
        right_arm = qpos[QPOS_RIGHT_ARM].astype(np.float32)
        left_gripper = proprioception_grippers[0:1].astype(np.float32)
        right_gripper = proprioception_grippers[1:2].astype(np.float32)
        torso = proprioception_floating_base[2:3].astype(np.float32)
        
        # Calculate base velocity from position diff
        if self.prev_floating_base is not None:
            # Velocity = (current - prev) at 50Hz
            vel_x = proprioception_floating_base[0] - self.prev_floating_base[0]
            vel_y = proprioception_floating_base[1] - self.prev_floating_base[1]
            vel_rz = proprioception_floating_base[3] - self.prev_floating_base[3]
            base_velocity = np.array([vel_x, vel_y, vel_rz], dtype=np.float32)
        else:
            base_velocity = np.zeros(3, dtype=np.float32)
        
        self.prev_floating_base = proprioception_floating_base.copy()
        
        result = {
            'base_velocity': base_velocity,
            'torso': torso,
            'left_arm': left_arm,
            'left_gripper': left_gripper,
            'right_arm': right_arm,
            'right_gripper': right_gripper,
        }
        
        if self.use_rgb:
            rgb = obs.get("rgb_head", np.zeros((3, 224, 224), dtype=np.uint8))
            result['rgb'] = rgb.astype(np.float32) / 255.0
        else:
            depth = obs.get("depth_head", np.zeros((224, 224), dtype=np.float32))
            rgb_img = obs.get("rgb_head", np.zeros((3, 224, 224), dtype=np.uint8))
            pcd_xyz, pcd_rgb = depth_to_pointcloud(depth, rgb_img, n_points=self.n_points)
            result['pcd_xyz'] = pcd_xyz
            result['pcd_rgb'] = pcd_rgb
        
        return result
    
    def _normalize_features(self, features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Apply normalization to features."""
        result = features.copy()
        
        # Normalize proprioception
        if self.normalize_prop and self.prop_min is not None:
            # Concatenate, normalize, split
            prop_concat = np.concatenate([
                features['base_velocity'],  # 3
                features['torso'],          # 1
                features['left_arm'],       # 5
                features['left_gripper'],   # 1
                features['right_arm'],      # 5
                features['right_gripper'],  # 1
            ])  # Total: 16
            
            prop_normalized = self._normalize_to_range(prop_concat, self.prop_min, self.prop_max)
            
            result['base_velocity'] = prop_normalized[0:3]
            result['torso'] = prop_normalized[3:4]
            result['left_arm'] = prop_normalized[4:9]
            result['left_gripper'] = prop_normalized[9:10]
            result['right_arm'] = prop_normalized[10:15]
            result['right_gripper'] = prop_normalized[15:16]
        
        # Normalize PCD
        if not self.use_rgb and self.normalize_pcd and self.pcd_xyz_min is not None:
            result['pcd_xyz'] = self._normalize_to_range(
                features['pcd_xyz'], self.pcd_xyz_min, self.pcd_xyz_max
            )
        
        return result
    
    def update(self, obs: Dict[str, Any]):
        """Update history with new observation."""
        features = self._extract_features(obs)
        features = self._normalize_features(features)
        
        # Add to history
        for key in features:
            if key in self.history:
                self.history[key].append(features[key])
                # Keep only last num_latest_obs
                if len(self.history[key]) > self.num_latest_obs:
                    self.history[key] = self.history[key][-self.num_latest_obs:]
    
    def get_observation(self) -> Dict[str, torch.Tensor]:
        """Get observation dict for policy."""
        # Pad history if not enough observations yet
        for key in self.history:
            while len(self.history[key]) < self.num_latest_obs:
                if self.history[key]:
                    # Repeat the first observation
                    self.history[key].insert(0, self.history[key][0])
                else:
                    # Use zeros
                    if key == 'base_velocity':
                        self.history[key].append(np.zeros(3, dtype=np.float32))
                    elif key == 'torso':
                        self.history[key].append(np.zeros(1, dtype=np.float32))
                    elif key in ['left_arm', 'right_arm']:
                        self.history[key].append(np.zeros(5, dtype=np.float32))
                    elif key in ['left_gripper', 'right_gripper']:
                        self.history[key].append(np.zeros(1, dtype=np.float32))
                    elif key == 'rgb':
                        self.history[key].append(np.zeros((3, 224, 224), dtype=np.float32))
                    elif key == 'pcd_xyz':
                        self.history[key].append(np.zeros((self.n_points, 3), dtype=np.float32))
                    elif key == 'pcd_rgb':
                        self.history[key].append(np.zeros((self.n_points, 3), dtype=np.float32))
        
        def to_tensor(arr_list):
            arr = np.stack(arr_list, axis=0)  # (T, ...)
            t = torch.from_numpy(arr).float()
            t = t.unsqueeze(0)  # Add batch dim: (1, T, ...)
            return t.to(self.device)
        
        obs_dict = {
            "odom": {"base_velocity": to_tensor(self.history['base_velocity'])},
            "qpos": {
                "torso": to_tensor(self.history['torso']),
                "left_arm": to_tensor(self.history['left_arm']),
                "left_gripper": to_tensor(self.history['left_gripper']),
                "right_arm": to_tensor(self.history['right_arm']),
                "right_gripper": to_tensor(self.history['right_gripper']),
            },
        }
        
        if self.use_rgb:
            # RGB policy expects tensor directly, not dict
            obs_dict["rgb"] = to_tensor(self.history['rgb'])
        else:
            obs_dict["pointcloud"] = {
                "xyz": to_tensor(self.history['pcd_xyz']),
                "rgb": to_tensor(self.history['pcd_rgb']),
            }
        
        return obs_dict


def create_environment(
    env_name: str,
    use_rgb: bool = False,
    render: bool = False,
    include_torso: bool = False,
) -> Any:
    """
    Create BigYM environment with 4-DOF floating base.
    
    Args:
        env_name: Environment name from ENV_REGISTRY
        use_rgb: Whether to include RGB observations
        render: Whether to render
        include_torso: If True, env has 17D actions (insert torso at index 4)
                      If False, env has 16D actions (matches demo)
    
    Returns:
        BigYM environment
    """
    if env_name not in ENV_REGISTRY:
        raise ValueError(f"Unknown environment: {env_name}. Available: {list(ENV_REGISTRY.keys())}")
    
    env_class = ENV_REGISTRY[env_name]
    
    # 4-DOF floating base (matches demo format)
    floating_dofs = [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
    
    # Observation config - always provide a config (even for PCD mode)
    if use_rgb:
        obs_config = ObservationConfig(
            cameras=[
                CameraConfig(name="head", rgb=True, depth=False, resolution=(224, 224)),
            ],
        )
    else:
        # For PCD mode, we still need depth images to create point clouds
        obs_config = ObservationConfig(
            cameras=[
                CameraConfig(name="head", rgb=True, depth=True, resolution=(224, 224)),
            ],
        )
    
    env = env_class(
        robot_cls=H1FineManipulation,
        action_mode=JointPositionActionMode(
            floating_base=True,
            floating_dofs=floating_dofs,
            absolute=True,  # Arms absolute, floating base still delta
        ),
        observation_config=obs_config,
        render_mode="human" if render else "rgb_array",
    )
    
    # Check if env includes torso
    env_action_dim = env.action_space.shape[0]
    if env_action_dim == 17:
        print(f"Environment has 17D action space (includes torso)")
    elif env_action_dim == 16:
        print(f"Environment has 16D action space (no torso)")
    else:
        print(f"Warning: Unexpected action dimension: {env_action_dim}")
    
    return env


def convert_brs_action_to_env(
    brs_action: np.ndarray,
    env_action_dim: int = 16,
) -> np.ndarray:
    """
    Convert BRS 16D action to environment action space.
    
    BRS/Demo Action (16D):
        [0:4]   floating_base (pelvis_x, pelvis_y, pelvis_z, pelvis_rz) - DELTA
        [4:14]  arms (10 joints) - ABSOLUTE
        [14:16] grippers (2) - COMMAND
    
    Environment Action:
        16D (torso=False): Same as BRS, no conversion needed
        17D (torso=True): Insert torso=0 at index 4
            [0:4]   floating_base
            [4]     torso (set to 0)
            [5:15]  arms
            [15:17] grippers
    """
    if brs_action.shape[-1] != 16:
        raise ValueError(f"Expected BRS action dim 16, got {brs_action.shape[-1]}")
    
    if env_action_dim == 16:
        # No conversion needed
        return brs_action
    
    elif env_action_dim == 17:
        # Insert torso at index 4
        squeeze = False
        if brs_action.ndim == 1:
            brs_action = brs_action[None, :]
            squeeze = True
        
        B = brs_action.shape[0]
        env_action = np.zeros((B, 17), dtype=brs_action.dtype)
        
        env_action[:, 0:4] = brs_action[:, 0:4]    # floating base
        env_action[:, 4] = 0.0                      # torso (not in demo)
        env_action[:, 5:15] = brs_action[:, 4:14]  # arms
        env_action[:, 15:17] = brs_action[:, 14:16] # grippers
        
        if squeeze:
            env_action = env_action[0]
        
        return env_action
    
    else:
        raise ValueError(f"Unexpected env action dim: {env_action_dim}")


def convert_action_for_500hz(
    action_50hz: np.ndarray,
    decimation: int = ENV_DECIMATION,
) -> np.ndarray:
    """
    Convert 50Hz policy action to 500Hz environment action.
    
    Policy outputs 50Hz actions (trained on decimated demo data).
    Environment runs at 500Hz internally.
    
    Floating base deltas in 50Hz = SUM of 10 x 500Hz deltas
    So we need to DIVIDE by decimation factor.
    
    Arms and grippers are absolute/command, no conversion needed.
    """
    action_500hz = action_50hz.copy()
    
    # Divide floating base deltas by decimation factor
    # Floating base is always at indices 0:4 (pelvis_x, pelvis_y, pelvis_z, pelvis_rz)
    if action_500hz.ndim == 1:
        action_500hz[0:4] = action_500hz[0:4] / decimation
    else:
        action_500hz[:, 0:4] = action_500hz[:, 0:4] / decimation
    
    return action_500hz


def prepare_observation(
    obs: Dict[str, Any],
    use_rgb: bool = False,
    history_len: int = 1,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Prepare BigYM observation for BRS policy input.
    
    BigYM observation structure:
        proprioception: (60,) - qpos(30) + qvel(30)
        proprioception_grippers: (2,)
        proprioception_floating_base: (4,) - [x, y, z, rz] absolute position
        rgb_head: (3, 224, 224) uint8
        depth_head: (224, 224) float32
    
    BRS policy expects:
        odom: {"base_velocity": (B, T, 3)}  # [vx, vy, vrz]
        qpos: {
            "torso": (B, T, 1),        # z position
            "left_arm": (B, T, 5),
            "left_gripper": (B, T, 1),
            "right_arm": (B, T, 5),
            "right_gripper": (B, T, 1),
        }
        pointcloud: {"xyz": (B, T, N, 3), "rgb": (B, T, N, 3)}  # for PCD mode
        OR
        rgb: {"head": (B, T, C, H, W)}  # for RGB mode
    """
    # Extract proprioception data from BigYM
    proprioception = obs.get("proprioception", np.zeros(60))  # qpos(30) + qvel(30)
    proprioception_grippers = obs.get("proprioception_grippers", np.zeros(2))
    proprioception_floating_base = obs.get("proprioception_floating_base", np.zeros(4))
    
    # Extract joint positions from proprioception (first 30 are qpos)
    qpos = proprioception[:30]
    
    # QPOS indices from README_HDF5_MAPPING.md:
    # qpos[0-3]:   left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow
    # qpos[4-11]:  left_gripper joints (8 joints)
    # qpos[12]:    left_wrist
    # qpos[13-16]: right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow
    # qpos[17-24]: right_gripper joints (8 joints)
    # qpos[25]:    right_wrist
    # qpos[26-29]: pelvis_x, pelvis_y, pelvis_z, pelvis_rz
    
    QPOS_LEFT_ARM = [0, 1, 2, 3, 12]       # 5D: shoulder_pitch, roll, yaw, elbow, wrist
    QPOS_RIGHT_ARM = [13, 14, 15, 16, 25]  # 5D: shoulder_pitch, roll, yaw, elbow, wrist
    
    left_arm = qpos[QPOS_LEFT_ARM]
    right_arm = qpos[QPOS_RIGHT_ARM]
    
    # Grippers from proprioception_grippers
    left_gripper = proprioception_grippers[0:1]
    right_gripper = proprioception_grippers[1:2]
    
    # Torso (z position) from floating base
    torso = proprioception_floating_base[2:3]  # pelvis_z
    
    # Base velocity - for now use zeros since we don't have velocity directly
    # In practice, this should be computed from diff of positions
    base_velocity = np.zeros(3, dtype=np.float32)  # [vx, vy, vrz]
    
    # Build observation dict
    def to_tensor(arr, add_batch_time=True):
        t = torch.from_numpy(np.array(arr, dtype=np.float32))
        if add_batch_time:
            t = t.unsqueeze(0).unsqueeze(0)  # Add (B=1, T=1) dims
        return t.to(device)
    
    if use_rgb:
        # Get RGB observation
        rgb = obs.get("rgb_head", np.zeros((3, 224, 224), dtype=np.uint8))
        
        # Convert to float [0, 1] and add batch/time dims
        rgb = torch.from_numpy(rgb).float() / 255.0
        rgb = rgb.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, 3, 224, 224)
        
        obs_dict = {
            "odom": {"base_velocity": to_tensor(base_velocity)},
            "qpos": {
                "torso": to_tensor(torso),
                "left_arm": to_tensor(left_arm),
                "left_gripper": to_tensor(left_gripper),
                "right_arm": to_tensor(right_arm),
                "right_gripper": to_tensor(right_gripper),
            },
            "rgb": {"head": rgb},
        }
    else:
        # Get depth for point cloud creation
        depth = obs.get("depth_head", np.zeros((224, 224), dtype=np.float32))
        rgb_img = obs.get("rgb_head", np.zeros((3, 224, 224), dtype=np.uint8))
        
        # Create point cloud from depth image
        pcd_xyz, pcd_rgb = depth_to_pointcloud(depth, rgb_img, n_points=4096)
        
        obs_dict = {
            "odom": {"base_velocity": to_tensor(base_velocity)},
            "qpos": {
                "torso": to_tensor(torso),
                "left_arm": to_tensor(left_arm),
                "left_gripper": to_tensor(left_gripper),
                "right_arm": to_tensor(right_arm),
                "right_gripper": to_tensor(right_gripper),
            },
            "pointcloud": {
                "xyz": to_tensor(pcd_xyz),
                "rgb": to_tensor(pcd_rgb),
            },
        }
    
    return obs_dict


def depth_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    n_points: int = 4096,
    fx: float = 320.0,
    fy: float = 320.0,
    cx: float = 112.0,
    cy: float = 112.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert depth image to point cloud.
    
    Args:
        depth: (H, W) depth image in meters
        rgb: (3, H, W) RGB image in uint8
        n_points: Number of points to sample
        fx, fy: Focal lengths
        cx, cy: Principal point
    
    Returns:
        xyz: (n_points, 3) point cloud coordinates
        rgb_pcd: (n_points, 3) point cloud colors (normalized to [0, 1])
    """
    H, W = depth.shape
    
    # Create pixel coordinates
    u = np.arange(W)
    v = np.arange(H)
    u, v = np.meshgrid(u, v)
    
    # Convert to 3D coordinates
    z = depth
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack to point cloud
    xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)
    
    # Get RGB values (convert from CHW to HWC and flatten)
    rgb_hwc = rgb.transpose(1, 2, 0)  # (H, W, 3)
    rgb_flat = rgb_hwc.reshape(-1, 3).astype(np.float32) / 255.0
    
    # Filter out invalid points (depth = 0 or too far)
    valid_mask = (z.flatten() > 0.01) & (z.flatten() < 10.0)
    xyz_valid = xyz[valid_mask]
    rgb_valid = rgb_flat[valid_mask]
    
    # Sample n_points
    if len(xyz_valid) > n_points:
        indices = np.random.choice(len(xyz_valid), n_points, replace=False)
        xyz_sampled = xyz_valid[indices]
        rgb_sampled = rgb_valid[indices]
    elif len(xyz_valid) > 0:
        # Pad with zeros if not enough points
        indices = np.random.choice(len(xyz_valid), n_points, replace=True)
        xyz_sampled = xyz_valid[indices]
        rgb_sampled = rgb_valid[indices]
    else:
        # Return zeros if no valid points
        xyz_sampled = np.zeros((n_points, 3), dtype=np.float32)
        rgb_sampled = np.zeros((n_points, 3), dtype=np.float32)
    
    return xyz_sampled.astype(np.float32), rgb_sampled.astype(np.float32)


@torch.no_grad()
def evaluate_episode(
    env,
    model,
    config: Dict[str, Any],
    stats: Dict[str, Any],
    use_rgb: bool = False,
    max_steps: int = 500,
    device: str = "cuda",
    save_video: bool = False,
    video_path: Optional[str] = None,
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Evaluate a single episode with proper decimation handling.
    
    Policy outputs 50Hz actions, environment runs at 500Hz.
    We divide floating base deltas by 10 and execute 10 steps per action.
    
    Args:
        seed: If provided, set this seed before resetting environment.
              Each episode can use a different seed for diverse evaluation.
    """
    # Set seed for this episode if provided
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
    
    obs, info = env.reset()
    env_action_dim = env.action_space.shape[0]
    
    # Create observation history manager
    num_latest_obs = config.get('num_latest_obs', 2)
    n_points = config.get('pcd_downsample_points', 4096)
    normalize_prop = config.get('normalize', False)
    normalize_pcd = config.get('normalize_pcd', False)
    
    obs_manager = ObservationHistoryManager(
        num_latest_obs=num_latest_obs,
        n_points=n_points,
        use_rgb=use_rgb,
        normalize_prop=normalize_prop,
        normalize_pcd=normalize_pcd,
        prop_stats=stats.get('prop_stats'),
        pcd_stats=stats.get('pcd_stats'),
        device=device,
    )
    
    # Initialize history with first observation
    obs_manager.update(obs)
    
    done = False
    truncated = False
    total_reward = 0.0
    step = 0
    action_count = 0
    
    video_frames = []
    action_stats = stats.get('action_stats')
    
    while not (done or truncated) and step < max_steps:
        # Get observation from history manager
        obs_dict = obs_manager.get_observation()
        
        # Get action from policy (50Hz)
        # model.policy.act returns dict with mobile_base, torso, arms
        # NOTE: Use float32 for inference to avoid NaN issues with float16
        with torch.no_grad():
            action_dict = model.policy.act(obs_dict)
        
        # Convert action dict to 16D numpy array
        # BRS format: [mobile_base(3), torso(1), arms(12)] = 16D
        mobile_base = action_dict["mobile_base"]  # (B, H, 3) or similar
        torso = action_dict["torso"]              # (B, H, 1)
        arms = action_dict["arms"]                # (B, H, 12)
        
        # DEBUG: Print shapes before extraction (only on first action of first episode)
        # if action_count == 0:
        #     print(f"DEBUG: num_latest_obs={num_latest_obs}, normalize_prop={normalize_prop}, normalize_pcd={normalize_pcd}")
        #     print(f"DEBUG: obs_dict keys: {list(obs_dict.keys())}")
        #     for k, v in obs_dict.items():
        #         if isinstance(v, dict):
        #             for k2, v2 in v.items():
        #                 print(f"  DEBUG: {k}/{k2} shape: {v2.shape if hasattr(v2, 'shape') else type(v2)}")
        #                 if hasattr(v2, 'min'):
        #                     print(f"    DEBUG: {k}/{k2} range: [{v2.min().item():.4f}, {v2.max().item():.4f}]")
        #                     if torch.isnan(v2).any():
        #                         print(f"    DEBUG: {k}/{k2} has NaN!")
        #         else:
        #             print(f"  DEBUG: {k} shape: {v.shape if hasattr(v, 'shape') else type(v)}")
        #     print(f"DEBUG: mobile_base shape: {mobile_base.shape}")
        #     print(f"DEBUG: torso shape: {torso.shape}")
        #     print(f"DEBUG: arms shape: {arms.shape}")
        
        # Take first timestep of action horizon
        if mobile_base.dim() >= 2:
            mobile_base = mobile_base[0, 0] if mobile_base.dim() == 3 else mobile_base[0]
        if torso.dim() >= 2:
            torso = torso[0, 0] if torso.dim() == 3 else torso[0]
        if arms.dim() >= 2:
            arms = arms[0, 0] if arms.dim() == 3 else arms[0]
        
        # Concatenate to 16D action (normalized)
        action_normalized = torch.cat([mobile_base, torso, arms], dim=-1).cpu().numpy()
        
        # Denormalize action if action_stats provided
        if action_stats is not None:
            action_50hz = denormalize_action(action_normalized, action_stats)
        else:
            action_50hz = action_normalized
        
        # Convert 16D BRS action to env action (16D or 17D)
        action_env = convert_brs_action_to_env(action_50hz, env_action_dim)
        
        # Convert 50Hz action to 500Hz (divide floating base by decimation)
        action_500hz = convert_action_for_500hz(action_env, ENV_DECIMATION)
        
        action_count += 1
        
        # Execute action for decimation steps (500Hz env)
        for _ in range(ENV_DECIMATION):
            # Clip to action space
            action_clipped = np.clip(action_500hz, env.action_space.low, env.action_space.high)
            
            # Step environment
            obs, reward, done, truncated, info = env.step(action_clipped)
            total_reward += reward
            step += 1
            
            # Save frame for video
            if save_video and HAS_CV2:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)
            
            if done or truncated or step >= max_steps:
                break
        
        # Update observation history (at 50Hz rate)
        if not (done or truncated):
            obs_manager.update(obs)
    
    # Save video if requested
    if save_video and HAS_CV2 and video_frames and video_path:
        save_video_frames(video_frames, video_path)
    
    return {
        "reward": total_reward,
        "steps": step,
        "success": info.get("success", done and total_reward > 0),
        "done": done,
        "truncated": truncated,
    }


def save_video_frames(frames, video_path: str, fps: int = 30):
    """Save frames as video using imageio (better compatibility) or OpenCV fallback."""
    if not frames:
        return
    
    # Use imageio if available (better codec support)
    if HAS_IMAGEIO:
        try:
            # imageio-ffmpeg provides better codec support
            writer = imageio.get_writer(
                video_path,
                fps=fps,
                codec='libx264',  # H.264 codec
                quality=8,  # 0-10, higher is better
                pixelformat='yuv420p',  # Better compatibility
            )
            for frame in frames:
                writer.append_data(frame)
            writer.close()
            print(f"Video saved to: {video_path}")
            return
        except Exception as e:
            print(f"Warning: imageio failed ({e}), trying OpenCV fallback")
    
    # OpenCV fallback
    if not HAS_CV2:
        print(f"Warning: Neither imageio nor cv2 available, cannot save video")
        return
    
    h, w = frames[0].shape[:2]
    
    # Try different codecs
    codecs_to_try = [
        ('avc1', '.mp4'),  # H.264
        ('H264', '.mp4'),  # H.264 alternative
        ('X264', '.mp4'),  # x264
        ('mp4v', '.mp4'),  # MPEG-4 fallback
    ]
    
    out = None
    for codec, ext in codecs_to_try:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        out = cv2.VideoWriter(video_path, fourcc, fps, (w, h))
        if out.isOpened():
            break
        out.release()
        out = None
    
    if out is None:
        print(f"Warning: Could not create video writer for {video_path}")
        return
    
    for frame in frames:
        # Convert RGB to BGR for OpenCV
        if frame.shape[-1] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame)
    
    out.release()
    print(f"Video saved to: {video_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate BRS policy on BigYM")
    
    # Checkpoint options
    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint", "-c", type=str, help="Path to checkpoint file")
    ckpt_group.add_argument("--exp-dir", "-e", type=str, help="Experiment directory (auto-find latest checkpoint)")
    
    # Environment options
    parser.add_argument("--env", type=str, default="saucepan_to_hob", 
                        help=f"Environment name. Available: {list(ENV_REGISTRY.keys())}")
    parser.add_argument("--num-episodes", "-n", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    
    # Observation mode
    parser.add_argument("--use-rgb", action="store_true", help="Use RGB observations instead of PCD")
    
    # Rendering options
    parser.add_argument("--render", action="store_true", help="Render environment")
    parser.add_argument("--save-video", action="store_true", help="Save episode videos")
    parser.add_argument("--video-dir", type=str, default="eval_videos", help="Directory to save videos")
    
    # Other options
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    # Set seed
    pl.seed_everything(args.seed)
    
    # Find checkpoint
    if args.exp_dir:
        checkpoint_path = find_latest_checkpoint(args.exp_dir)
        if checkpoint_path is None:
            print(f"Error: No checkpoint found in {args.exp_dir}")
            return
        print(f"Found checkpoint: {checkpoint_path}")
    else:
        checkpoint_path = args.checkpoint
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            return
    
    # Load model
    print(f"Loading model from: {checkpoint_path}")
    model, config, stats = create_policy(checkpoint_path, use_rgb=args.use_rgb, device=args.device)
    print(f"Model loaded successfully. Use RGB: {args.use_rgb}")
    
    # Create environment
    print(f"Creating environment: {args.env}")
    env = create_environment(args.env, use_rgb=args.use_rgb, render=args.render)
    print(f"Environment action space: {env.action_space.shape}")
    
    # Create video directory with checkpoint-specific subfolder
    video_save_dir = None
    if args.save_video:
        # Extract experiment name from checkpoint path for hierarchical organization
        exp_name = Path(checkpoint_path).parent.parent.name
        obs_mode = "rgb" if args.use_rgb else "pcd"
        video_save_dir = os.path.join(args.video_dir, exp_name, obs_mode)
        os.makedirs(video_save_dir, exist_ok=True)
        print(f"Videos will be saved to: {video_save_dir}")
    
    # Evaluate with different seeds for each episode
    results = []
    pbar = tqdm(range(args.num_episodes), desc="Evaluating")
    
    for episode_idx in pbar:
        video_path = None
        if args.save_video and video_save_dir:
            video_path = os.path.join(video_save_dir, f"episode_{episode_idx:03d}.mp4")
        
        # Use different seed for each episode: base_seed + episode_idx
        episode_seed = args.seed + episode_idx
        
        result = evaluate_episode(
            env=env,
            model=model,
            config=config,
            stats=stats,
            use_rgb=args.use_rgb,
            max_steps=args.max_steps,
            device=args.device,
            save_video=args.save_video,
            video_path=video_path,
            seed=episode_seed,
        )
        result["seed"] = episode_seed  # Store seed in result
        results.append(result)
        
        # Update progress bar
        success_rate = sum(r["success"] for r in results) / len(results)
        avg_reward = sum(r["reward"] for r in results) / len(results)
        pbar.set_postfix({
            "success": f"{success_rate:.1%}",
            "reward": f"{avg_reward:.2f}",
        })
    
    # Print summary
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    
    success_rate = sum(r["success"] for r in results) / len(results)
    avg_reward = sum(r["reward"] for r in results) / len(results)
    avg_steps = sum(r["steps"] for r in results) / len(results)
    
    print(f"Environment: {args.env}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Num Episodes: {args.num_episodes}")
    print(f"Observation Mode: {'RGB' if args.use_rgb else 'PCD'}")
    print("-" * 50)
    print(f"Success Rate: {success_rate:.1%} ({sum(r['success'] for r in results)}/{len(results)})")
    print(f"Average Reward: {avg_reward:.3f}")
    print(f"Average Steps: {avg_steps:.1f}")
    print("=" * 50)
    
    env.close()


if __name__ == "__main__":
    main()
