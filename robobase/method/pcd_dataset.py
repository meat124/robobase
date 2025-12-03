"""
PCD Dataset for BRS Policy Training

Uses BigYM native 16D action format for direct environment interaction.

BigYM 16D Action Structure:
    - action[0:4]   = floating_base delta [dx, dy, dz, drz] - DELTA mode
    - action[4:9]   = left_arm (5D) - ABSOLUTE position
    - action[9:14]  = right_arm (5D) - ABSOLUTE position
    - action[14:16] = grippers (2D) - ABSOLUTE [0, 1]

BRS Policy 3-part autoregressive structure (reordered for prediction):
    - mobile_base: 3D [dx, dy, drz] - from floating_base[0,1,3]
    - torso: 1D [dz] - from floating_base[2]
    - arms: 12D [left_arm(5) + right_arm(5) + grippers(2)]

Proprioception 16D:
    - mobile_base_pos: 3D [x, y, rz] - from floating_base[0,1,3]
    - torso: 1D [z] - from floating_base[2]
    - left_arm: 5D, left_gripper: 1D, right_arm: 5D, right_gripper: 1D
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


class PCDBRSDataset(Dataset):
    """
    Dataset for BRS Policy training with BigYM native format.
    
    Loads HDF5 demonstrations with separate PCD files.
    Outputs data in BRS 3-part autoregressive format.
    """
    
    def __init__(
        self,
        hdf5_path: str,
        pcd_root: str,
        demo_ids: List[str],
        cameras: List[str] = ["head"],
        num_latest_obs: int = 2,
        action_prediction_horizon: int = 8,
        max_points: int = 4096,
        subsample_points: bool = True,
        action_stats_path: Optional[str] = None,
        prop_stats_path: Optional[str] = None,
        pcd_stats_path: Optional[str] = None,
        normalize: bool = True,
        normalize_pcd: bool = True,
        preload_hdf5: bool = False,
        preload_pcd: bool = False,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.pcd_root = Path(pcd_root) if pcd_root else None
        self.demo_ids = demo_ids
        self.cameras = cameras
        self.num_latest_obs = num_latest_obs
        self.action_prediction_horizon = action_prediction_horizon
        self.max_points = max_points
        self.subsample_points = subsample_points
        self.normalize = normalize
        self.normalize_pcd = normalize_pcd
        self.preload_hdf5 = preload_hdf5
        self.preload_pcd = preload_pcd
        
        # Load normalization statistics
        self._load_stats(action_stats_path, prop_stats_path, pcd_stats_path)
        
        # Build sample index
        self.samples = self._build_sample_index()
        
        # HDF5 file handle (lazy loading per worker)
        self._hdf5_file = None
        
        # Preloaded data caches
        self._preloaded_hdf5 = None
        self._pcd_cache = {}
        
        # Preload data if requested
        if preload_hdf5:
            self._preload_hdf5_data()
        if preload_pcd:
            self._preload_pcd_data()
        
        print(f"PCDBRSDataset initialized:")
        print(f"  - HDF5: {self.hdf5_path}")
        print(f"  - PCD root: {self.pcd_root}")
        print(f"  - Demos: {len(demo_ids)}")
        print(f"  - Total samples: {len(self.samples)}")
    
    def _load_stats(self, action_stats_path, prop_stats_path, pcd_stats_path):
        """Load normalization statistics from JSON files."""
        if not self.normalize:
            self.action_min = self.action_max = None
            self.prop_min = self.prop_max = None
            self.pcd_xyz_min = self.pcd_xyz_max = None
            return
        
        # Default paths (same directory as HDF5)
        stats_dir = self.hdf5_path.parent
        action_stats_path = action_stats_path or stats_dir / "action_stats.json"
        prop_stats_path = prop_stats_path or stats_dir / "prop_stats.json"
        pcd_stats_path = pcd_stats_path or stats_dir / "pcd_stats.json"
        
        # Load action stats
        self.action_min, self.action_max = self._load_stats_file(
            action_stats_path, "action", 
            component_keys=['mobile_base', 'torso', 'arms']
        )
        
        # Load proprioception stats
        self.prop_min, self.prop_max = self._load_stats_file(
            prop_stats_path, "prop",
            component_keys=['mobile_base_pos', 'torso', 'arms']
        )
        
        # Load PCD stats
        if Path(pcd_stats_path).exists():
            with open(pcd_stats_path, 'r') as f:
                pcd_stats = json.load(f)
            self.pcd_xyz_min = np.array(pcd_stats['xyz']['min'], dtype=np.float32)
            self.pcd_xyz_max = np.array(pcd_stats['xyz']['max'], dtype=np.float32)
            print(f"  Loaded PCD stats")
        else:
            print(f"  Warning: PCD stats not found at {pcd_stats_path}")
            self.pcd_xyz_min = self.pcd_xyz_max = None
    
    def _load_stats_file(self, path, name, component_keys):
        """Load stats from JSON file with 'full' or component format."""
        if not Path(path).exists():
            print(f"  Warning: {name} stats not found at {path}")
            return None, None
        
        with open(path, 'r') as f:
            stats = json.load(f)
        
        # Prefer 'full' key
        if 'full' in stats:
            min_val = np.array(stats['full']['min'], dtype=np.float32)
            max_val = np.array(stats['full']['max'], dtype=np.float32)
        elif all(k in stats for k in component_keys):
            min_val, max_val = [], []
            for k in component_keys:
                v_min = stats[k]['min']
                v_max = stats[k]['max']
                if isinstance(v_min, list):
                    min_val.extend(v_min)
                    max_val.extend(v_max)
                else:
                    min_val.append(v_min)
                    max_val.append(v_max)
            min_val = np.array(min_val, dtype=np.float32)
            max_val = np.array(max_val, dtype=np.float32)
        else:
            print(f"  Warning: Unrecognized {name} stats format")
            return None, None
        
        print(f"  Loaded {name} stats: shape={min_val.shape}")
        return min_val, max_val
    
    def _build_sample_index(self) -> List[Tuple[str, int, int]]:
        """Build index of valid samples: (demo_id, start_frame, num_frames)."""
        samples = []
        
        with h5py.File(self.hdf5_path, 'r') as f:
            for demo_id in self.demo_ids:
                if demo_id not in f:
                    print(f"  Warning: {demo_id} not found in HDF5")
                    continue
                    
                demo_data = f[demo_id]
                num_frames = demo_data['actions'].shape[0]
                
                # Check PCD availability
                if self.pcd_root:
                    demo_idx = int(demo_id.split('_')[1])
                    npy_path = self.pcd_root / f"demo_{demo_idx:03d}_pcd.npy"
                    if not npy_path.exists():
                        print(f"  Warning: PCD not found for {demo_id}")
                        continue
                
                # Create samples with sliding window
                min_frames = self.num_latest_obs + self.action_prediction_horizon
                for start_idx in range(max(0, num_frames - min_frames + 1)):
                    samples.append((demo_id, start_idx, num_frames))
        
        return samples
    
    def _preload_hdf5_data(self):
        """Preload all HDF5 data into RAM."""
        print("  Preloading HDF5 data into RAM...")
        self._preloaded_hdf5 = {}
        
        with h5py.File(self.hdf5_path, 'r') as f:
            for demo_id in self.demo_ids:
                if demo_id not in f:
                    continue
                
                demo_data = f[demo_id]
                self._preloaded_hdf5[demo_id] = {
                    'actions': demo_data['actions'][:],
                    'proprioception': demo_data['proprioception'][:],
                    'proprioception_floating_base': demo_data['proprioception_floating_base'][:],
                    'proprioception_grippers': demo_data['proprioception_grippers'][:],
                }
        
        total_bytes = sum(
            arr.nbytes for data in self._preloaded_hdf5.values() for arr in data.values()
        )
        print(f"  Preloaded HDF5: {len(self._preloaded_hdf5)} demos, {total_bytes / 1024 / 1024:.1f} MB")
    
    def _preload_pcd_data(self):
        """Preload all PCD data into RAM."""
        if self.pcd_root is None:
            return
        
        print("  Preloading PCD data into RAM...")
        total_bytes = 0
        
        for demo_id in self.demo_ids:
            demo_idx = int(demo_id.split('_')[1])
            npy_path = self.pcd_root / f"demo_{demo_idx:03d}_pcd.npy"
            
            if npy_path.exists():
                pcd_data = np.load(npy_path)
                self._pcd_cache[demo_id] = pcd_data
                total_bytes += pcd_data.nbytes
        
        print(f"  Preloaded PCD: {len(self._pcd_cache)} demos, {total_bytes / 1024 / 1024:.1f} MB")
    
    def _get_hdf5_file(self):
        """Get HDF5 file handle."""
        if self._preloaded_hdf5 is not None:
            return None
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(self.hdf5_path, 'r')
        return self._hdf5_file
    
    def _get_demo_data(self, demo_id: str):
        """Get demo data from preloaded cache or HDF5 file."""
        if self._preloaded_hdf5 is not None and demo_id in self._preloaded_hdf5:
            return self._preloaded_hdf5[demo_id]
        return self._get_hdf5_file()[demo_id]
    
    def _normalize_to_range(self, value: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
        """Normalize value to [-1, 1] range."""
        range_val = np.maximum(max_val - min_val, 1e-8)
        normalized = 2.0 * (value - min_val) / range_val - 1.0
        return np.clip(normalized, -1.0, 1.0)
    
    def _extract_proprioception(self, demo_data, obs_indices) -> Dict[str, np.ndarray]:
        """Extract proprioception in BRS format."""
        if isinstance(demo_data, dict):
            prop = demo_data['proprioception'][obs_indices]
            fb = demo_data['proprioception_floating_base'][obs_indices]
            grippers = demo_data['proprioception_grippers'][obs_indices]
        else:
            prop = demo_data['proprioception'][obs_indices]
            fb = demo_data['proprioception_floating_base'][obs_indices]
            grippers = demo_data['proprioception_grippers'][obs_indices]
        
        return {
            'mobile_base_pos': np.column_stack([fb[:, 0], fb[:, 1], fb[:, 3]]).astype(np.float32),
            'torso': fb[:, 2:3].astype(np.float32),
            'left_arm': np.column_stack([prop[:, i] for i in QPOS_LEFT_ARM]).astype(np.float32),
            'left_gripper': grippers[:, 0:1].astype(np.float32),
            'right_arm': np.column_stack([prop[:, i] for i in QPOS_RIGHT_ARM]).astype(np.float32),
            'right_gripper': grippers[:, 1:2].astype(np.float32),
        }
    
    def _process_actions(self, demo_data, action_indices) -> Dict[str, np.ndarray]:
        """Extract actions in BRS format."""
        if isinstance(demo_data, dict):
            actions = demo_data['actions'][action_indices]
        else:
            actions = demo_data['actions'][action_indices]
        
        return {
            'mobile_base': np.column_stack([actions[:, 0], actions[:, 1], actions[:, 3]]).astype(np.float32),
            'torso': actions[:, 2:3].astype(np.float32),
            'arms': actions[:, 4:16].astype(np.float32),
        }
    
    def _load_pcd_frame(self, demo_id: str, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load PCD from files."""
        if demo_id in self._pcd_cache:
            pcd_all = self._pcd_cache[demo_id]
            if frame_idx < len(pcd_all):
                xyz = pcd_all[frame_idx]
                return xyz, np.zeros_like(xyz)
        
        demo_idx = int(demo_id.split('_')[1])
        npy_path = self.pcd_root / f"demo_{demo_idx:03d}_pcd.npy"
        
        if npy_path.exists():
            pcd_all = np.load(npy_path)
            if frame_idx < len(pcd_all):
                xyz = pcd_all[frame_idx]
                return xyz, np.zeros_like(xyz)
        
        return np.zeros((self.max_points, 3), dtype=np.float32), np.zeros((self.max_points, 3), dtype=np.float32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        demo_id, start_frame, num_frames = self.samples[idx]
        demo_data = self._get_demo_data(demo_id)
        
        # Observation and action indices
        obs_indices = list(range(start_frame, start_frame + self.num_latest_obs))
        action_start = start_frame + self.num_latest_obs - 1
        action_end = min(action_start + self.action_prediction_horizon, num_frames)
        action_indices = list(range(action_start, action_end))
        
        n_actions = len(action_indices)
        pad_length = self.action_prediction_horizon - n_actions
        
        # Extract data
        prop_data = self._extract_proprioception(demo_data, obs_indices)
        action_data = self._process_actions(demo_data, action_indices)
        
        # Pad actions if needed
        if pad_length > 0:
            for key in action_data:
                pad_shape = (pad_length,) + action_data[key].shape[1:]
                action_data[key] = np.concatenate([action_data[key], np.zeros(pad_shape, dtype=np.float32)], axis=0)
        
        # Normalize proprioception
        if self.normalize and self.prop_min is not None:
            prop_concat = np.concatenate([
                prop_data['mobile_base_pos'], prop_data['torso'],
                prop_data['left_arm'], prop_data['left_gripper'],
                prop_data['right_arm'], prop_data['right_gripper'],
            ], axis=-1)
            prop_concat = self._normalize_to_range(prop_concat, self.prop_min, self.prop_max)
            
            prop_data['mobile_base_pos'] = prop_concat[:, 0:3]
            prop_data['torso'] = prop_concat[:, 3:4]
            prop_data['left_arm'] = prop_concat[:, 4:9]
            prop_data['left_gripper'] = prop_concat[:, 9:10]
            prop_data['right_arm'] = prop_concat[:, 10:15]
            prop_data['right_gripper'] = prop_concat[:, 15:16]
        
        # Normalize actions
        if self.normalize and self.action_min is not None:
            action_concat = np.concatenate([
                action_data['mobile_base'], action_data['torso'], action_data['arms']
            ], axis=-1)
            action_concat = self._normalize_to_range(action_concat, self.action_min, self.action_max)
            
            action_data['mobile_base'] = action_concat[:, 0:3]
            action_data['torso'] = action_concat[:, 3:4]
            action_data['arms'] = action_concat[:, 4:16]
        
        # Load point clouds
        pcd_xyz = np.zeros((self.num_latest_obs, self.max_points, 3), dtype=np.float32)
        pcd_rgb = np.zeros((self.num_latest_obs, self.max_points, 3), dtype=np.float32)
        
        if self.pcd_root:
            for t, frame_idx in enumerate(obs_indices):
                xyz, rgb = self._load_pcd_frame(demo_id, frame_idx)
                n = min(len(xyz), self.max_points)
                if n > 0:
                    pcd_xyz[t, :n] = xyz[:n]
                    pcd_rgb[t, :n] = rgb[:n]
        
        # Normalize PCD
        if self.normalize_pcd and self.pcd_xyz_min is not None:
            pcd_xyz = self._normalize_to_range(pcd_xyz, self.pcd_xyz_min, self.pcd_xyz_max)
        
        # Padding mask
        pad_mask = np.ones((self.num_latest_obs, self.action_prediction_horizon), dtype=np.float32)
        if pad_length > 0:
            pad_mask[:, -pad_length:] = 0.0
        
        # Tile actions for each observation timestep
        action_chunks = {
            'mobile_base': np.tile(action_data['mobile_base'][None, :, :], (self.num_latest_obs, 1, 1)),
            'torso': np.tile(action_data['torso'][None, :, :], (self.num_latest_obs, 1, 1)),
            'arms': np.tile(action_data['arms'][None, :, :], (self.num_latest_obs, 1, 1)),
        }
        
        return {
            'qpos': {
                'mobile_base_pos': prop_data['mobile_base_pos'],
                'torso': prop_data['torso'],
                'left_arm': prop_data['left_arm'],
                'left_gripper': prop_data['left_gripper'],
                'right_arm': prop_data['right_arm'],
                'right_gripper': prop_data['right_gripper'],
            },
            'pointcloud': {'xyz': pcd_xyz, 'rgb': pcd_rgb},
            'action_chunks': action_chunks,
            'pad_mask': pad_mask,
        }


def pcd_brs_collate_fn(batch):
    """Collate function for BRS dataset."""
    odom_mobile_base_pos = torch.from_numpy(np.stack([b["qpos"]["mobile_base_pos"] for b in batch]))
    qpos_torso = torch.from_numpy(np.stack([b["qpos"]["torso"] for b in batch]))
    qpos_left_arm = torch.from_numpy(np.stack([b["qpos"]["left_arm"] for b in batch]))
    qpos_left_gripper = torch.from_numpy(np.stack([b["qpos"]["left_gripper"] for b in batch]))
    qpos_right_arm = torch.from_numpy(np.stack([b["qpos"]["right_arm"] for b in batch]))
    qpos_right_gripper = torch.from_numpy(np.stack([b["qpos"]["right_gripper"] for b in batch]))
    pcd_xyz = torch.from_numpy(np.stack([b["pointcloud"]["xyz"] for b in batch]))
    pcd_rgb = torch.from_numpy(np.stack([b["pointcloud"]["rgb"] for b in batch]))
    
    action_chunks = {
        "mobile_base": torch.from_numpy(np.stack([b["action_chunks"]["mobile_base"] for b in batch])),
        "torso": torch.from_numpy(np.stack([b["action_chunks"]["torso"] for b in batch])),
        "arms": torch.from_numpy(np.stack([b["action_chunks"]["arms"] for b in batch])),
    }
    pad_mask = torch.from_numpy(np.stack([b["pad_mask"] for b in batch]))
    
    # Add N_chunks dimension: (B, ...) -> (1, B, ...)
    return {
        "odom": {"mobile_base_pos": odom_mobile_base_pos.unsqueeze(0)},
        "qpos": {
            "torso": qpos_torso.unsqueeze(0),
            "left_arm": qpos_left_arm.unsqueeze(0),
            "left_gripper": qpos_left_gripper.unsqueeze(0),
            "right_arm": qpos_right_arm.unsqueeze(0),
            "right_gripper": qpos_right_gripper.unsqueeze(0),
        },
        "pointcloud": {"xyz": pcd_xyz.unsqueeze(0), "rgb": pcd_rgb.unsqueeze(0)},
        "action_chunks": {k: v.unsqueeze(0) for k, v in action_chunks.items()},
        "pad_mask": pad_mask.unsqueeze(0),
    }


class PCDDataModule(pl.LightningDataModule):
    """DataModule for PCD-based BRS training."""
    
    def __init__(
        self,
        hdf5_path: str,
        pcd_root: str,
        demo_ids: Optional[List[str]] = None,
        cameras: List[str] = ["head"],
        num_latest_obs: int = 2,
        action_prediction_horizon: int = 8,
        max_points_per_camera: int = 4096,
        batch_size: int = 8,
        val_batch_size: int = 16,
        val_split_ratio: float = 0.1,
        dataloader_num_workers: int = 4,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        seed: Optional[int] = None,
        normalize: bool = True,
        action_stats_path: Optional[str] = None,
        prop_stats_path: Optional[str] = None,
        pcd_stats_path: Optional[str] = None,
        normalize_pcd: bool = True,
        subsample_points: bool = True,
        preload_hdf5: bool = False,
        preload_pcd: bool = False,
    ):
        super().__init__()
        self.hdf5_path = Path(hdf5_path)
        self.pcd_root = Path(pcd_root) if pcd_root else None
        self.demo_ids = demo_ids
        self.cameras = cameras
        self.num_latest_obs = num_latest_obs
        self.action_prediction_horizon = action_prediction_horizon
        self.max_points = max_points_per_camera
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.val_split_ratio = val_split_ratio
        self.dataloader_num_workers = dataloader_num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.seed = seed
        self.normalize = normalize
        self.action_stats_path = action_stats_path
        self.prop_stats_path = prop_stats_path
        self.pcd_stats_path = pcd_stats_path
        self.normalize_pcd = normalize_pcd
        self.subsample_points = subsample_points
        self.preload_hdf5 = preload_hdf5
        self.preload_pcd = preload_pcd
        
        if self.demo_ids is None:
            self.demo_ids = self._discover_demo_ids()
    
    def _discover_demo_ids(self) -> List[str]:
        """Discover all demo IDs from HDF5 file."""
        with h5py.File(self.hdf5_path, 'r') as f:
            return [k for k in f.keys() if k.startswith('demo_')]
    
    def setup(self, stage: Optional[str] = None):
        """Setup train/val datasets."""
        all_demos = sorted(self.demo_ids)
        n_val = max(1, int(len(all_demos) * self.val_split_ratio))
        
        if self.seed is not None:
            np.random.seed(self.seed)
        np.random.shuffle(all_demos)
        
        val_demos = all_demos[:n_val]
        train_demos = all_demos[n_val:]
        
        print(f"\nDataset split:")
        print(f"  Train demos ({len(train_demos)}): {train_demos[:5]}...")
        print(f"  Val demos ({len(val_demos)}): {val_demos}")
        
        dataset_kwargs = dict(
            hdf5_path=str(self.hdf5_path),
            pcd_root=str(self.pcd_root) if self.pcd_root else None,
            cameras=self.cameras,
            num_latest_obs=self.num_latest_obs,
            action_prediction_horizon=self.action_prediction_horizon,
            max_points=self.max_points,
            subsample_points=self.subsample_points,
            action_stats_path=self.action_stats_path,
            prop_stats_path=self.prop_stats_path,
            pcd_stats_path=self.pcd_stats_path,
            normalize=self.normalize,
            normalize_pcd=self.normalize_pcd,
            preload_hdf5=self.preload_hdf5,
            preload_pcd=self.preload_pcd,
        )
        
        self.train_dataset = PCDBRSDataset(demo_ids=train_demos, **dataset_kwargs)
        self.val_dataset = PCDBRSDataset(demo_ids=val_demos, **dataset_kwargs)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
            prefetch_factor=self.prefetch_factor if self.dataloader_num_workers > 0 else None,
            persistent_workers=self.persistent_workers and self.dataloader_num_workers > 0,
            collate_fn=pcd_brs_collate_fn,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
            prefetch_factor=self.prefetch_factor if self.dataloader_num_workers > 0 else None,
            persistent_workers=self.persistent_workers and self.dataloader_num_workers > 0,
            collate_fn=pcd_brs_collate_fn,
            pin_memory=True,
        )
