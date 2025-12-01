"""
RGB Dataset for BRS Policy Training (RGB version replacing PCD).

Designed for the following HDF5 structure (BigYM SaucepanToHob):
    demo_N/
        actions: (T, 16) float32
        proprioception: (T+1, 60) float32  # [qpos(30) + qvel(30)]
        proprioception_floating_base: (T+1, 4) float32  # [x, y, z, rz] absolute
        proprioception_floating_base_actions: (T+1, 4) float32  # [dx, dy, dz, drz] delta
        proprioception_grippers: (T+1, 2) float32  # [left, right]
        rgb_head: (T+1, 3, 224, 224) uint8
        rgb_left_wrist: (T+1, 3, 224, 224) uint8 (optional)
        rgb_right_wrist: (T+1, 3, 224, 224) uint8 (optional)

BigYM 16D Action Structure:
    - action[0:4]   = floating_base delta [dx, dy, dz, drz] - DELTA mode
    - action[4:9]   = left_arm (5D) - ABSOLUTE position
    - action[9:14]  = right_arm (5D) - ABSOLUTE position
    - action[14:16] = grippers (2D) - ABSOLUTE [0, 1]

BRS Policy expects 16D actions in 3-part autoregressive structure:
    - mobile_base: 3D (x_vel, y_vel, rz_vel) - from action[0,1,3]/dt
    - torso: 1D (z position delta) - from action[2]
    - arms: 12D (left_arm 5 + left_gripper 1 + right_arm 5 + right_gripper 1)
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import json


# qpos indices for BigYM H1 robot
QPOS_LEFT_ARM = [0, 1, 2, 3, 12]       # 5D: shoulder_pitch, roll, yaw, elbow, wrist
QPOS_RIGHT_ARM = [13, 14, 15, 16, 25]  # 5D: shoulder_pitch, roll, yaw, elbow, wrist
QPOS_LEFT_GRIPPER = 4   # left gripper driver joint
QPOS_RIGHT_GRIPPER = 17  # right gripper driver joint


class RGBBRSDataset(Dataset):
    """
    Dataset for BRS Policy training with RGB images.
    
    Handles BigYM SaucepanToHob HDF5 format with 16D actions.
    
    Supports preloading data into RAM for faster training.
    """
    
    def __init__(
        self,
        hdf5_path: str,
        demo_ids: List[str],
        cameras: List[str] = ["head"],
        num_latest_obs: int = 2,
        action_prediction_horizon: int = 8,
        image_size: Tuple[int, int] = (224, 224),
        action_stats_path: Optional[str] = None,
        prop_stats_path: Optional[str] = None,
        normalize: bool = True,
        preload_data: bool = False,
    ):
        """
        Args:
            hdf5_path: Path to HDF5 file (demos.hdf5)
            demo_ids: List of demo IDs to use (e.g., ['demo_0', 'demo_1', ...])
            cameras: Camera names for RGB loading (default: ['head'])
            num_latest_obs: Temporal window size for observations
            action_prediction_horizon: Number of future actions to predict
            image_size: Expected image size (H, W)
            action_stats_path: Path to action statistics JSON
            prop_stats_path: Path to proprioception statistics JSON  
            normalize: Whether to normalize actions/proprioception to [-1, 1]
            preload_data: Preload all data into RAM (faster training)
        """
        self.hdf5_path = Path(hdf5_path)
        self.demo_ids = demo_ids
        self.cameras = cameras
        self.num_latest_obs = num_latest_obs
        self.action_prediction_horizon = action_prediction_horizon
        self.image_size = image_size
        self.normalize = normalize
        self.preload_data = preload_data
        self.dt = 0.02  # 50Hz control
        
        # RGB keys from camera names
        self.rgb_keys = [f"rgb_{cam}" for cam in cameras]
        
        # Load normalization statistics
        self._load_stats(action_stats_path, prop_stats_path)
        
        # Build sample index
        self.samples = self._build_sample_index()
        
        # HDF5 file handle (lazy loading per worker)
        self._hdf5_file = None
        
        # Preloaded data caches
        self._preloaded_data = None
        
        # Preload data if requested
        if preload_data:
            self._preload_all_data()
        
        print(f"RGBBRSDataset initialized:")
        print(f"  - HDF5: {self.hdf5_path}")
        print(f"  - Cameras: {self.cameras}")
        print(f"  - Demos: {len(demo_ids)}")
        print(f"  - Total samples: {len(self.samples)}")
        print(f"  - Window size: {num_latest_obs}")
        print(f"  - Action horizon: {action_prediction_horizon}")
        print(f"  - Normalize: {normalize}")
        print(f"  - Preload: {preload_data}")
    
    def _load_stats(self, action_stats_path, prop_stats_path):
        """Load normalization statistics from JSON files."""
        if not self.normalize:
            self.action_min = self.action_max = None
            self.prop_min = self.prop_max = None
            return
        
        # Default paths (same directory as HDF5)
        stats_dir = self.hdf5_path.parent
        action_stats_path = action_stats_path or stats_dir / "action_stats.json"
        prop_stats_path = prop_stats_path or stats_dir / "prop_stats.json"
        
        # Load action stats
        if Path(action_stats_path).exists():
            with open(action_stats_path, 'r') as f:
                action_stats = json.load(f)
            
            # Check for 'full' key (new format with 16D BRS format)
            if 'full' in action_stats:
                self.action_min = np.array(action_stats['full']['min'], dtype=np.float32)
                self.action_max = np.array(action_stats['full']['max'], dtype=np.float32)
                print(f"  Loaded action stats (full): shape={self.action_min.shape}")
            else:
                raise ValueError("action_stats.json must have 'full' key with 16D BRS format")
        else:
            raise FileNotFoundError(f"Action stats not found: {action_stats_path}")
        
        # Load proprioception stats  
        if Path(prop_stats_path).exists():
            with open(prop_stats_path, 'r') as f:
                prop_stats = json.load(f)
            
            if 'full' in prop_stats:
                self.prop_min = np.array(prop_stats['full']['min'], dtype=np.float32)
                self.prop_max = np.array(prop_stats['full']['max'], dtype=np.float32)
                print(f"  Loaded prop stats (full): shape={self.prop_min.shape}")
            else:
                raise ValueError("prop_stats.json must have 'full' key with 16D format")
        else:
            raise FileNotFoundError(f"Prop stats not found: {prop_stats_path}")
    
    def _build_sample_index(self) -> List[Tuple[str, int]]:
        """Build index of valid (demo_id, timestep) samples."""
        samples = []
        
        with h5py.File(self.hdf5_path, 'r') as f:
            for demo_id in self.demo_ids:
                if demo_id not in f:
                    print(f"Warning: {demo_id} not found in HDF5, skipping")
                    continue
                
                demo = f[demo_id]
                
                # Get action length (T)
                action_length = demo['actions'].shape[0]
                
                # Valid starting indices: need num_latest_obs-1 frames before
                # and action_prediction_horizon frames after
                start_idx = self.num_latest_obs - 1
                end_idx = action_length - self.action_prediction_horizon + 1
                
                for t in range(start_idx, end_idx):
                    samples.append((demo_id, t))
        
        return samples
    
    def _preload_all_data(self):
        """Preload all HDF5 data into RAM."""
        import time
        start_time = time.time()
        
        self._preloaded_data = {}
        total_frames = 0
        
        print("Preloading RGB data into memory...")
        
        with h5py.File(self.hdf5_path, 'r') as f:
            for demo_idx, demo_id in enumerate(self.demo_ids):
                if demo_id not in f:
                    continue
                
                demo = f[demo_id]
                
                self._preloaded_data[demo_id] = {
                    'actions': demo['actions'][:],
                    'proprioception': demo['proprioception'][:],
                    'proprioception_floating_base': demo['proprioception_floating_base'][:],
                    'proprioception_grippers': demo['proprioception_grippers'][:],
                }
                
                # Load RGB for each camera
                for rgb_key in self.rgb_keys:
                    if rgb_key in demo:
                        self._preloaded_data[demo_id][rgb_key] = demo[rgb_key][:]
                        total_frames += demo[rgb_key].shape[0]
                
                if (demo_idx + 1) % 10 == 0:
                    print(f"  Preloaded {demo_idx + 1}/{len(self.demo_ids)} demos")
        
        elapsed = time.time() - start_time
        # Estimate memory: RGB frames * 3 * 224 * 224 bytes
        memory_gb = total_frames * 3 * 224 * 224 / (1024**3)
        print(f"Preload complete: {total_frames} frames, ~{memory_gb:.2f}GB RGB, {elapsed:.1f}s")
    
    def _get_hdf5_file(self):
        """Get HDF5 file handle (lazy initialization for worker processes)."""
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r')
        return self._hdf5_file
    
    def __len__(self):
        return len(self.samples)
    
    def _extract_proprioception_16d(
        self,
        qpos: np.ndarray,
        floating_base: np.ndarray,
        grippers: np.ndarray,
    ) -> np.ndarray:
        """Extract 16D proprioception: [base_vel(3) + torso(1) + arms(10) + grippers(2)]"""
        # Base velocity (from floating_base delta)
        # floating_base is absolute [x, y, z, rz], we use as-is for now
        base_xy = floating_base[:2]  # [x, y]
        base_rz = floating_base[3:4]  # [rz]
        base_vel = np.concatenate([base_xy, base_rz])  # [3]
        
        # Torso (z position)
        torso = floating_base[2:3]  # [z] -> [1]
        
        # Left arm (5D)
        left_arm = qpos[QPOS_LEFT_ARM]  # [5]
        
        # Left gripper (1D)
        left_gripper = grippers[0:1]  # [1]
        
        # Right arm (5D)
        right_arm = qpos[QPOS_RIGHT_ARM]  # [5]
        
        # Right gripper (1D)
        right_gripper = grippers[1:2]  # [1]
        
        # Concatenate: [3 + 1 + 5 + 1 + 5 + 1] = 16
        prop_16d = np.concatenate([
            base_vel, torso, left_arm, left_gripper, right_arm, right_gripper
        ])
        
        return prop_16d.astype(np.float32)
    
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1]."""
        if self.action_min is None:
            return action
        
        # Min-max normalization to [-1, 1]
        action_range = self.action_max - self.action_min
        action_range = np.maximum(action_range, 1e-6)  # Avoid div by zero
        
        normalized = 2.0 * (action - self.action_min) / action_range - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
    
    def _normalize_proprioception(self, prop: np.ndarray) -> np.ndarray:
        """Normalize proprioception to [-1, 1]."""
        if self.prop_min is None:
            return prop
        
        prop_range = self.prop_max - self.prop_min
        prop_range = np.maximum(prop_range, 1e-6)
        
        normalized = 2.0 * (prop - self.prop_min) / prop_range - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
    
    def _bigym_action_to_brs(self, bigym_action: np.ndarray) -> np.ndarray:
        """
        Convert BigYM 16D action to BRS 16D format.
        
        BigYM: [fb(4), left_arm(5), right_arm(5), grippers(2)]
        BRS:   [mobile_base(3), torso(1), arms(12)]
        
        arms = [left_arm(5), left_gripper(1), right_arm(5), right_gripper(1)]
        """
        # Floating base: [dx, dy, dz, drz] -> mobile_base[dx, dy, drz], torso[dz]
        mobile_base = np.array([bigym_action[0], bigym_action[1], bigym_action[3]], dtype=np.float32)
        torso = np.array([bigym_action[2]], dtype=np.float32)
        
        # Arms: [left_arm(5), right_arm(5), grippers(2)] -> [left_arm(5), left_grip(1), right_arm(5), right_grip(1)]
        left_arm = bigym_action[4:9]
        right_arm = bigym_action[9:14]
        left_gripper = bigym_action[14:15]
        right_gripper = bigym_action[15:16]
        
        arms = np.concatenate([left_arm, left_gripper, right_arm, right_gripper])
        
        # BRS format: [mobile_base(3), torso(1), arms(12)]
        brs_action = np.concatenate([mobile_base, torso, arms])
        
        return brs_action.astype(np.float32)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        demo_id, timestep = self.samples[idx]
        
        # Get data source (preloaded or HDF5)
        if self._preloaded_data is not None and demo_id in self._preloaded_data:
            data = self._preloaded_data[demo_id]
        else:
            f = self._get_hdf5_file()
            data = f[demo_id]
        
        # Get observation window indices
        obs_indices = list(range(timestep - self.num_latest_obs + 1, timestep + 1))
        
        # Get action window indices
        action_indices = list(range(timestep, timestep + self.action_prediction_horizon))
        
        # ===== Load RGB images =====
        # Shape: (T, C, H, W) per camera, combine to (T, V, C, H, W) if multi-view
        rgb_list = []
        for rgb_key in self.rgb_keys:
            if self._preloaded_data is not None:
                rgb = data[rgb_key][obs_indices]  # (T, C, H, W)
            else:
                rgb = data[rgb_key][obs_indices]  # (T, C, H, W)
            rgb_list.append(rgb)
        
        if len(rgb_list) == 1:
            # Single camera: (T, C, H, W)
            rgb = rgb_list[0]
        else:
            # Multi-camera: (T, V, C, H, W)
            rgb = np.stack(rgb_list, axis=1)
        
        # Convert to float [0, 1]
        rgb = rgb.astype(np.float32) / 255.0
        
        # ===== Load proprioception =====
        prop_obs_list = []
        for t in obs_indices:
            if self._preloaded_data is not None:
                qpos_qvel = data['proprioception'][t]
                floating_base = data['proprioception_floating_base'][t]
                grippers = data['proprioception_grippers'][t]
            else:
                qpos_qvel = data['proprioception'][t]
                floating_base = data['proprioception_floating_base'][t]
                grippers = data['proprioception_grippers'][t]
            
            qpos = qpos_qvel[:30]  # First 30 are qpos
            prop_16d = self._extract_proprioception_16d(qpos, floating_base, grippers)
            prop_obs_list.append(prop_16d)
        
        prop_obs = np.stack(prop_obs_list, axis=0)  # (T, 16)
        
        # ===== Load actions =====
        action_list = []
        for t in action_indices:
            if self._preloaded_data is not None:
                bigym_action = data['actions'][t]
            else:
                bigym_action = data['actions'][t]
            
            brs_action = self._bigym_action_to_brs(bigym_action)
            action_list.append(brs_action)
        
        actions = np.stack(action_list, axis=0)  # (H, 16)
        
        # ===== Normalize =====
        if self.normalize:
            prop_obs = self._normalize_proprioception(prop_obs)
            actions = np.stack([self._normalize_action(a) for a in actions], axis=0)
        
        # ===== Split proprioception into BRS format =====
        # prop_obs: (T, 16) -> split into components
        prop_data = {
            'mobile_base_vel': prop_obs[:, 0:3],    # (T, 3)
            'torso': prop_obs[:, 3:4],              # (T, 1)
            'left_arm': prop_obs[:, 4:9],           # (T, 5)
            'left_gripper': prop_obs[:, 9:10],      # (T, 1)
            'right_arm': prop_obs[:, 10:15],        # (T, 5)
            'right_gripper': prop_obs[:, 15:16],    # (T, 1)
        }
        
        # ===== Split actions into BRS format =====
        # actions: (H, 16) -> split into [mobile_base(3), torso(1), arms(12)]
        action_data = {
            'mobile_base': actions[:, 0:3],   # (H, 3)
            'torso': actions[:, 3:4],         # (H, 1)
            'arms': actions[:, 4:16],         # (H, 12)
        }
        
        # ===== Create action_chunks (tiled for each obs timestep) =====
        # BRS format expects: action_chunks[key] = (T, H, dim) for each sample
        action_chunks = {
            'mobile_base': np.tile(action_data['mobile_base'][None, :, :], (self.num_latest_obs, 1, 1)),  # (T, H, 3)
            'torso': np.tile(action_data['torso'][None, :, :], (self.num_latest_obs, 1, 1)),              # (T, H, 1)
            'arms': np.tile(action_data['arms'][None, :, :], (self.num_latest_obs, 1, 1)),                # (T, H, 12)
        }
        
        # ===== Create padding mask =====
        pad_mask = np.ones((self.num_latest_obs, self.action_prediction_horizon), dtype=np.float32)
        
        return {
            'rgb': rgb,  # (T, C, H, W) or (T, V, C, H, W) - numpy array
            'odom': {
                'base_velocity': prop_data['mobile_base_vel'],  # (T, 3)
            },
            'qpos': {
                'torso': prop_data['torso'],           # (T, 1)
                'left_arm': prop_data['left_arm'],     # (T, 5)
                'left_gripper': prop_data['left_gripper'],  # (T, 1)
                'right_arm': prop_data['right_arm'],   # (T, 5)
                'right_gripper': prop_data['right_gripper'],  # (T, 1)
            },
            'action_chunks': action_chunks,  # dict with (T, H, dim) arrays
            'pad_mask': pad_mask,  # (T, H)
        }

    
    def __del__(self):
        if self._hdf5_file is not None:
            self._hdf5_file.close()


def rgb_brs_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate function for RGBBRSDataset.
    
    Returns data in brs-algo format with N_chunks dimension:
    - All tensors have shape (N_chunks=1, B, ...)
    """
    import numpy as np
    
    # Stack RGB images
    rgb = torch.from_numpy(np.stack([b["rgb"] for b in batch]))  # (B, T, C, H, W)
    
    # Stack proprioception components
    odom_base_velocity = torch.from_numpy(np.stack([b["odom"]["base_velocity"] for b in batch]))
    qpos_torso = torch.from_numpy(np.stack([b["qpos"]["torso"] for b in batch]))
    qpos_left_arm = torch.from_numpy(np.stack([b["qpos"]["left_arm"] for b in batch]))
    qpos_left_gripper = torch.from_numpy(np.stack([b["qpos"]["left_gripper"] for b in batch]))
    qpos_right_arm = torch.from_numpy(np.stack([b["qpos"]["right_arm"] for b in batch]))
    qpos_right_gripper = torch.from_numpy(np.stack([b["qpos"]["right_gripper"] for b in batch]))
    
    # Stack action chunks
    action_chunks = {
        "mobile_base": torch.from_numpy(np.stack([b["action_chunks"]["mobile_base"] for b in batch])),
        "torso": torch.from_numpy(np.stack([b["action_chunks"]["torso"] for b in batch])),
        "arms": torch.from_numpy(np.stack([b["action_chunks"]["arms"] for b in batch])),
    }
    
    # Stack padding mask
    pad_mask = torch.from_numpy(np.stack([b["pad_mask"] for b in batch]))
    
    # Add N_chunks dimension (N_chunks=1)
    # Shape: (B, ...) -> (N_chunks=1, B, ...)
    return {
        "rgb": rgb.unsqueeze(0),  # (1, B, T, C, H, W)
        "odom": {
            "base_velocity": odom_base_velocity.unsqueeze(0),  # (1, B, T, 3)
        },
        "qpos": {
            "torso": qpos_torso.unsqueeze(0),  # (1, B, T, 1)
            "left_arm": qpos_left_arm.unsqueeze(0),  # (1, B, T, 5)
            "left_gripper": qpos_left_gripper.unsqueeze(0),  # (1, B, T, 1)
            "right_arm": qpos_right_arm.unsqueeze(0),  # (1, B, T, 5)
            "right_gripper": qpos_right_gripper.unsqueeze(0),  # (1, B, T, 1)
        },
        "action_chunks": {k: v.unsqueeze(0) for k, v in action_chunks.items()},  # (1, B, T, H, dim)
        "pad_mask": pad_mask.unsqueeze(0),  # (1, B, T, H)
    }



class RGBDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for RGB BRS Dataset.
    """
    
    def __init__(
        self,
        hdf5_path: str,
        cameras: List[str] = ["head"],
        num_latest_obs: int = 2,
        action_prediction_horizon: int = 8,
        image_size: Tuple[int, int] = (224, 224),
        action_stats_path: Optional[str] = None,
        prop_stats_path: Optional[str] = None,
        normalize: bool = True,
        batch_size: int = 64,
        val_batch_size: int = 64,
        num_workers: int = 8,
        val_split_ratio: float = 0.1,
        preload_data: bool = False,
    ):
        super().__init__()
        
        self.hdf5_path = hdf5_path
        self.cameras = cameras
        self.num_latest_obs = num_latest_obs
        self.action_prediction_horizon = action_prediction_horizon
        self.image_size = image_size
        self.action_stats_path = action_stats_path
        self.prop_stats_path = prop_stats_path
        self.normalize = normalize
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers
        self.val_split_ratio = val_split_ratio
        self.preload_data = preload_data
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        """Setup train/val datasets."""
        # Discover all demo IDs from HDF5
        with h5py.File(self.hdf5_path, 'r') as f:
            all_demo_ids = [k for k in f.keys() if k.startswith('demo_')]
        
        # Sort by demo number
        all_demo_ids.sort(key=lambda x: int(x.split('_')[1]))
        
        # Train/val split
        n_demos = len(all_demo_ids)
        n_val = max(1, int(n_demos * self.val_split_ratio))
        n_train = n_demos - n_val
        
        train_demo_ids = all_demo_ids[:n_train]
        val_demo_ids = all_demo_ids[n_train:]
        
        print(f"Dataset split: {n_train} train demos, {n_val} val demos")
        
        self.train_dataset = RGBBRSDataset(
            hdf5_path=self.hdf5_path,
            demo_ids=train_demo_ids,
            cameras=self.cameras,
            num_latest_obs=self.num_latest_obs,
            action_prediction_horizon=self.action_prediction_horizon,
            image_size=self.image_size,
            action_stats_path=self.action_stats_path,
            prop_stats_path=self.prop_stats_path,
            normalize=self.normalize,
            preload_data=self.preload_data,
        )
        
        self.val_dataset = RGBBRSDataset(
            hdf5_path=self.hdf5_path,
            demo_ids=val_demo_ids,
            cameras=self.cameras,
            num_latest_obs=self.num_latest_obs,
            action_prediction_horizon=self.action_prediction_horizon,
            image_size=self.image_size,
            action_stats_path=self.action_stats_path,
            prop_stats_path=self.prop_stats_path,
            normalize=self.normalize,
            preload_data=self.preload_data,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=rgb_brs_collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=rgb_brs_collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )
