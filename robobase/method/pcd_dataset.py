"""
PCD Dataset for BRS Policy Training
Loads point cloud data from separate PCD files and proprioception/actions from HDF5
"""

import h5py
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

class PCDBRSDataset(Dataset):
    """
    Dataset that loads point clouds, proprioception, and actions from HDF5
    
    HDF5 Structure:
        data/
            demo_N/
                actions: (num_frames, 16)
                obs/
                    proprioception: (num_frames, 60)
                    proprioception_floating_base: (num_frames, 6)
                observations/
                    point_cloud/
                        head: (num_frames, max_points, 6)  # xyz + rgb
                        left_wrist: (num_frames, max_points, 6)
                        right_wrist: (num_frames, max_points, 6)
    """
    
    def __init__(
        self,
        hdf5_path: str,
        pcd_root: str,
        demo_ids: List[str],
        cameras: List[str] = ["head", "left_wrist", "right_wrist"],
        num_latest_obs: int = 2,
        action_prediction_horizon: int = 10,
        max_points_per_camera: int = 2048,
        subsample_points: bool = True,
    ):
        """
        Args:
            hdf5_path: Path to HDF5 file with proprioception and actions
            pcd_root: Root directory containing PCD files
            demo_ids: List of demo IDs to use (e.g., ["demo_0", "demo_1", ...])
            cameras: List of camera names to load
            num_latest_obs: Number of latest observations in temporal window
            action_prediction_horizon: Number of future actions to predict
            max_points_per_camera: Maximum number of points per camera (for memory efficiency)
            subsample_points: Whether to randomly subsample points
        """
        self.hdf5_path = Path(hdf5_path)
        self.pcd_root = Path(pcd_root)
        self.demo_ids = demo_ids
        self.cameras = cameras
        self.num_latest_obs = num_latest_obs
        self.action_prediction_horizon = action_prediction_horizon
        self.max_points_per_camera = max_points_per_camera
        self.subsample_points = subsample_points
        
        # HDF5 file handle (will be opened per worker)
        self._hdf5_file = None
        
        # Build index of valid samples
        self.samples = self._build_sample_index()
        
        print(f"PCDBRSDataset initialized:")
        print(f"  - Total samples: {len(self.samples)}")
        print(f"  - Demos: {len(demo_ids)}")
        print(f"  - Cameras: {cameras}")
        print(f"  - Window size: {num_latest_obs}")
        print(f"  - Action horizon: {action_prediction_horizon}")
    
    def _get_hdf5_file(self):
        """
        Get HDF5 file handle. Opens file lazily and keeps it open for efficiency.
        Each worker process will have its own file handle.
        Uses optimized settings for fast reading.
        """
        if self._hdf5_file is None:
            # Open with optimized settings for read performance
            # - rdcc_nbytes: Read cache size (100MB per worker)
            # - rdcc_nslots: Number of chunk slots in cache
            # - swmr: Single Writer Multiple Reader mode
            self._hdf5_file = h5py.File(
                self.hdf5_path, 
                'r',
                rdcc_nbytes=100*1024*1024,  # 100MB cache
                rdcc_nslots=10000,  # 10k slots
                swmr=True  # Enable SWMR for concurrent access
            )
        return self._hdf5_file
    
    def _build_sample_index(self) -> List[Tuple[str, int, int]]:
        """
        Build index of valid samples: (demo_id, start_frame, end_frame)
        
        Returns:
            List of (demo_id, start_frame, num_frames)
        """
        samples = []
        
        with h5py.File(self.hdf5_path, 'r') as f:
            for demo_id in self.demo_ids:
                if demo_id not in f['data']:
                    print(f"Warning: {demo_id} not found in HDF5, skipping")
                    continue
                
                # Get number of frames from actions
                num_frames = f['data'][demo_id]['actions'][()].shape[0]
                
                # Check if point cloud data exists in HDF5
                demo_group = f['data'][demo_id]
                if 'observations' not in demo_group or 'point_cloud' not in demo_group['observations']:
                    print(f"Warning: Point cloud data not found for {demo_id}, skipping")
                    continue
                
                # Check if PCD directory exists
                demo_pcd_dir = self.pcd_root / demo_id
                if not demo_pcd_dir.exists():
                    print(f"Warning: PCD directory not found for {demo_id}, skipping")
                    continue
                
                # Create samples with sliding window
                # Each sample needs num_latest_obs frames for observation
                # and action_prediction_horizon frames for actions
                for start_idx in range(num_frames - self.num_latest_obs - self.action_prediction_horizon + 1):
                    samples.append((demo_id, start_idx, num_frames))
        
        return samples
    
    def _load_pcd_frame(self, demo_id: str, camera: str, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single PCD frame from file
        
        Returns:
            xyz: (N, 3) point coordinates
            rgb: (N, 3) point colors (0-1 range)
        """
        pcd_path = self.pcd_root / demo_id / camera / f"frame_{frame_idx:05d}.pcd"
        
        if not pcd_path.exists():
            # Return empty pointcloud if file doesn't exist
            print(f"Warning: PCD file not found: {pcd_path}")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        
        # Load PCD
        pcd = o3d.io.read_point_cloud(str(pcd_path))
        
        # Get points and colors
        xyz = np.asarray(pcd.points, dtype=np.float32)
        
        if pcd.has_colors():
            rgb = np.asarray(pcd.colors, dtype=np.float32)
        else:
            rgb = np.ones_like(xyz) * 0.5  # Gray color if no RGB
        
        # Subsample if needed
        if self.subsample_points and len(xyz) > self.max_points_per_camera:
            indices = np.random.choice(len(xyz), self.max_points_per_camera, replace=False)
            xyz = xyz[indices]
            rgb = rgb[indices]
        
        return xyz, rgb
    
    def _load_multi_camera_pcd(self, demo_id: str, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and combine point clouds from all cameras for a single frame
        
        Returns:
            xyz: (N_total, 3) combined point coordinates
            rgb: (N_total, 3) combined point colors
        """
        all_xyz = []
        all_rgb = []
        
        for camera in self.cameras:
            xyz, rgb = self._load_pcd_frame(demo_id, camera, frame_idx)
            if len(xyz) > 0:
                all_xyz.append(xyz)
                all_rgb.append(rgb)
        
        if len(all_xyz) == 0:
            # Return single point at origin if no data
            return np.zeros((1, 3), dtype=np.float32), np.ones((1, 3), dtype=np.float32) * 0.5
        
        # Concatenate all cameras
        xyz_combined = np.concatenate(all_xyz, axis=0)
        rgb_combined = np.concatenate(all_rgb, axis=0)
        
        return xyz_combined, rgb_combined
        
        return samples
    
    def _load_pcd_frame_from_hdf5(self, demo_group, camera: str, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a single point cloud frame from HDF5
        
        Args:
            demo_group: Open HDF5 demo group
            camera: Camera name
            frame_idx: Frame index
        
        Returns:
            xyz: (N, 3) point coordinates
            rgb: (N, 3) point colors (0-1 range)
        """
        try:
            # Load from HDF5: data/demo_N/observations/point_cloud/camera_name
            pcd_data = demo_group['observations']['point_cloud'][camera][frame_idx]
            
            # Split xyz and rgb
            xyz = pcd_data[:, :3].astype(np.float32)
            rgb = pcd_data[:, 3:6].astype(np.float32)
            
            # Subsample if needed
            if self.subsample_points and len(xyz) > self.max_points_per_camera:
                indices = np.random.choice(len(xyz), self.max_points_per_camera, replace=False)
                xyz = xyz[indices]
                rgb = rgb[indices]
            
            return xyz, rgb
            
        except Exception as e:
            print(f"Warning: Failed to load point cloud {camera}/frame_{frame_idx}: {e}")
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
    
    def _load_multi_camera_pcd_batch(self, demo_group, frame_indices: List[int]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and combine point clouds from all cameras for multiple frames at once.
        This is more efficient than loading frames one by one.
        
        Args:
            demo_group: Open HDF5 demo group
            frame_indices: List of frame indices to load
        
        Returns:
            xyz_batch: (n_frames, max_points, 3) padded point coordinates
            rgb_batch: (n_frames, max_points, 3) padded point colors
        """
        n_frames = len(frame_indices)
        max_points_fixed = self.max_points_per_camera * len(self.cameras)
        
        xyz_batch = np.zeros((n_frames, max_points_fixed, 3), dtype=np.float32)
        rgb_batch = np.zeros((n_frames, max_points_fixed, 3), dtype=np.float32)
        
        for t, frame_idx in enumerate(frame_indices):
            # Load all cameras for this frame
            all_xyz = []
            all_rgb = []
            
            for camera in self.cameras:
                xyz, rgb = self._load_pcd_frame_from_hdf5(demo_group, camera, frame_idx)
                if len(xyz) > 0:
                    all_xyz.append(xyz)
                    all_rgb.append(rgb)
            
            if len(all_xyz) > 0:
                # Concatenate all cameras
                xyz_combined = np.concatenate(all_xyz, axis=0)
                rgb_combined = np.concatenate(all_rgb, axis=0)
                
                # Pad/crop to fixed size
                n = min(len(xyz_combined), max_points_fixed)
                xyz_batch[t, :n] = xyz_combined[:n]
                rgb_batch[t, :n] = rgb_combined[:n]
        
        return xyz_batch, rgb_batch
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        demo_id, start_frame, num_frames = self.samples[idx]
        
        # Use persistent HDF5 file handle (CRITICAL FOR SPEED!)
        f = self._get_hdf5_file()
        demo_data = f['data'][demo_id]
        
        # Load proprioception for observation window
        obs_indices = range(start_frame, start_frame + self.num_latest_obs)
        
        # Joint positions (proprioception) - 16D total
        # Total 60D = 30D qpos + 30D qvel
        # qpos [0-29]: pelvis(4) + left_arm(5) + left_gripper(8) + right_arm(5) + right_gripper(8)
        # qvel [30-59]: same structure as qpos
        proprioception = demo_data['obs']['proprioception']
        prop_data = proprioception[obs_indices]  # (num_latest_obs, 60)
        gripper = demo_data['obs']['proprioception_grippers'][obs_indices]
        
        # Extract from qpos (0-29) - POSITION ONLY
        torso = prop_data[:, 2:3]                  # [2]: pelvis_z (1D)
        left_arm = prop_data[:, 4:9]               # [4-8]: left arm 5 joints (5D)
        right_arm = prop_data[:, 17:22]            # [17-21]: right arm 5 joints (5D)

        # Extract gripper positions from separate gripper data
        # Shape must be (num_latest_obs, 1) to match other components
        left_gripper = gripper[:, 0:1]             # left gripper (1D) - keep dimension
        right_gripper = gripper[:, 1:2]            # right gripper (1D) - keep dimension


        # Mobile base velocity from qvel (30-59) - VELOCITY ONLY (3D)
        mobile_base_vel = np.concatenate([
            prop_data[:, 30:32],   # [30-31]: x_vel, y_vel
            prop_data[:, 33:34],   # [33]: rz_vel (skip index 32 which is pelvis_z_vel)
        ], axis=-1)  # (num_latest_obs, 3)
        
        # Total 16D: torso(1) + left_arm(5) + left_gripper(1) + right_arm(5) + right_gripper(1) + mobile_base_vel(3)
        
        # Load actions for prediction horizon
        action_indices = range(start_frame, start_frame + self.action_prediction_horizon)
        actions = demo_data['actions'][action_indices]
        
        mobile_base_actions = actions[:, 0:3].astype(np.float32)  # 3 dims
        torso_actions = actions[:, 3:4].astype(np.float32)        # 1 dim
        arms_actions = actions[:, 4:16].astype(np.float32)        # 12 dims
        
        action_chunks = {
            "mobile_base": mobile_base_actions,
            "torso": torso_actions,
            "arms": arms_actions,
        }
        
        # Load point clouds for observation window from PCD files
        pcd_xyz_list = []
        pcd_rgb_list = []
        
        for frame_idx in obs_indices:
            xyz, rgb = self._load_multi_camera_pcd(demo_id, frame_idx)
            pcd_xyz_list.append(xyz)
            pcd_rgb_list.append(rgb)
        
        # Determine fixed max_points for all frames in this sample
        max_points_fixed = self.max_points_per_camera * len(self.cameras)
        
        # Pad/crop to fixed size for batching
        pcd_xyz_padded = np.zeros((self.num_latest_obs, max_points_fixed, 3), dtype=np.float32)
        pcd_rgb_padded = np.zeros((self.num_latest_obs, max_points_fixed, 3), dtype=np.float32)
        
        for t, (xyz, rgb) in enumerate(zip(pcd_xyz_list, pcd_rgb_list)):
            n = min(len(xyz), max_points_fixed)
            pcd_xyz_padded[t, :n] = xyz[:n]
            pcd_rgb_padded[t, :n] = rgb[:n]
        
        # Create observation dict
        # 16D proprioception: torso(1) + left_arm(5) + left_gripper(1) + right_arm(5) + right_gripper(1) + mobile_base_vel(3)
        obs = {
            "odom": {
                "base_velocity": mobile_base_vel.astype(np.float32),  # (num_latest_obs, 3)
            },
            "qpos": {
                "torso": torso.astype(np.float32),                    # (num_latest_obs, 1)
                "left_arm": left_arm.astype(np.float32),              # (num_latest_obs, 5)
                "left_gripper": left_gripper.astype(np.float32),      # (num_latest_obs, 1)
                "right_arm": right_arm.astype(np.float32),            # (num_latest_obs, 5)
                "right_gripper": right_gripper.astype(np.float32),    # (num_latest_obs, 1)
            },
            "pointcloud": {
                "xyz": pcd_xyz_padded,
                "rgb": pcd_rgb_padded,
            }
        }
        
        # Expand action_chunks to match observation window size (replicate for each obs)
        # Shape: (num_latest_obs, action_prediction_horizon, dim)
        # No padding needed - using BigYM native dimensions: mobile_base(3), torso(1), arms(12)
        action_chunks = {
            "mobile_base": np.tile(mobile_base_actions[None, :, :], (self.num_latest_obs, 1, 1)),  # (T_obs, T_act, 3)
            "torso": np.tile(torso_actions[None, :, :], (self.num_latest_obs, 1, 1)),              # (T_obs, T_act, 1)
            "arms": np.tile(arms_actions[None, :, :], (self.num_latest_obs, 1, 1)),                # (T_obs, T_act, 12)
        }
        
        # Padding mask (all valid)
        pad_mask = np.ones((self.num_latest_obs, self.action_prediction_horizon), dtype=np.float32)
        
        return {
            **obs,
            "action_chunks": action_chunks,
            "pad_mask": pad_mask,
        }


def pcd_brs_collate_fn(batch):
    """
    Collate function for PCD BRS dataset
    Returns data in brs-algo format with N_chunks dimension
    """
    # Stack all fields
    odom_base_velocity = torch.from_numpy(np.stack([b["odom"]["base_velocity"] for b in batch]))
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
    
    # Add N_chunks dimension (we use 1 chunk for simplicity)
    # Shape: (B, window_size, ...) -> (N_chunks=1, B, window_size, ...)
    return {
        "odom": {
            "base_velocity": odom_base_velocity.unsqueeze(0),  # (1, B, window_size, 3)
        },
        "qpos": {
            "torso": qpos_torso.unsqueeze(0),
            "left_arm": qpos_left_arm.unsqueeze(0),
            "left_gripper": qpos_left_gripper.unsqueeze(0),
            "right_arm": qpos_right_arm.unsqueeze(0),
            "right_gripper": qpos_right_gripper.unsqueeze(0),
        },
        "pointcloud": {
            "xyz": pcd_xyz.unsqueeze(0),  # (1, B, window_size, n_points, 3)
            "rgb": pcd_rgb.unsqueeze(0),
        },
        "action_chunks": {k: v.unsqueeze(0) for k, v in action_chunks.items()},
        "pad_mask": pad_mask.unsqueeze(0),
    }


class PCDDataModule(pl.LightningDataModule):
    """
    DataModule for PCD-based BRS training
    """
    
    def __init__(
        self,
        hdf5_path: str,
        pcd_root: str,
        demo_ids: Optional[List[str]] = None,
        cameras: List[str] = ["head", "left_wrist", "right_wrist"],
        num_latest_obs: int = 2,
        action_prediction_horizon: int = 10,
        max_points_per_camera: int = 2048,
        batch_size: int = 8,
        val_batch_size: int = 16,
        val_split_ratio: float = 0.1,
        dataloader_num_workers: int = 4,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.pcd_root = pcd_root
        self.demo_ids = demo_ids
        self.cameras = cameras
        self.num_latest_obs = num_latest_obs
        self.action_prediction_horizon = action_prediction_horizon
        self.max_points_per_camera = max_points_per_camera
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.val_split_ratio = val_split_ratio
        self.dataloader_num_workers = dataloader_num_workers
        self.seed = seed
        
        # Auto-discover demo_ids if not provided
        if self.demo_ids is None:
            self.demo_ids = self._discover_demo_ids()
    
    def _discover_demo_ids(self) -> List[str]:
        """Auto-discover available demo IDs from PCD directory"""
        pcd_root = Path(self.pcd_root)
        demo_dirs = sorted([d for d in pcd_root.iterdir() if d.is_dir() and d.name.startswith('demo_')])
        demo_ids = [d.name for d in demo_dirs]
        print(f"Auto-discovered {len(demo_ids)} demos from PCD directory: {demo_ids}")
        return demo_ids
    
    def setup(self, stage: Optional[str] = None):
        # Split demos into train/val
        np.random.seed(self.seed if self.seed else 42)
        demo_ids_shuffled = np.random.permutation(self.demo_ids).tolist()
        
        num_val = max(1, int(len(demo_ids_shuffled) * self.val_split_ratio))
        val_demo_ids = demo_ids_shuffled[:num_val]
        train_demo_ids = demo_ids_shuffled[num_val:]
        
        print(f"\nDataset split:")
        print(f"  Train demos ({len(train_demo_ids)}): {train_demo_ids}")
        print(f"  Val demos ({len(val_demo_ids)}): {val_demo_ids}")
        
        # Create datasets
        self.train_dataset = PCDBRSDataset(
            hdf5_path=self.hdf5_path,
            pcd_root=self.pcd_root,
            demo_ids=train_demo_ids,
            cameras=self.cameras,
            num_latest_obs=self.num_latest_obs,
            action_prediction_horizon=self.action_prediction_horizon,
            max_points_per_camera=self.max_points_per_camera,
            subsample_points=True,
        )
        
        self.val_dataset = PCDBRSDataset(
            hdf5_path=self.hdf5_path,
            pcd_root=self.pcd_root,
            demo_ids=val_demo_ids,
            cameras=self.cameras,
            num_latest_obs=self.num_latest_obs,
            action_prediction_horizon=self.action_prediction_horizon,
            max_points_per_camera=self.max_points_per_camera,
            subsample_points=False,  # No random subsampling for validation
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
            collate_fn=pcd_brs_collate_fn,
            persistent_workers=self.dataloader_num_workers > 0,
            pin_memory=True,
            prefetch_factor=2 if self.dataloader_num_workers > 0 else None,  # Prefetch 2 batches per worker
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
            collate_fn=pcd_brs_collate_fn,
            persistent_workers=self.dataloader_num_workers > 0,
            pin_memory=True,
            prefetch_factor=2 if self.dataloader_num_workers > 0 else None,
        )
