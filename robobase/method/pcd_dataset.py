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
import json
import os

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
        action_stats_path: Optional[str] = None,
        prop_stats_path: Optional[str] = None,
        pcd_stats_path: Optional[str] = None,
        normalize: bool = True,
        normalize_pcd: bool = True,
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
            action_stats_path: Path to action statistics JSON file
            prop_stats_path: Path to proprioception statistics JSON file
            pcd_stats_path: Path to PCD XYZ statistics JSON file
            normalize: Whether to normalize actions and proprioception to [-1, 1]
            normalize_pcd: Whether to normalize PCD XYZ to [-1, 1] (set False if already preprocessed)
        """
        self.hdf5_path = Path(hdf5_path)
        self.pcd_root = Path(pcd_root)
        self.demo_ids = demo_ids
        self.cameras = cameras
        self.num_latest_obs = num_latest_obs
        self.action_prediction_horizon = action_prediction_horizon
        self.max_points_per_camera = max_points_per_camera
        self.subsample_points = subsample_points
        self.normalize = normalize
        self.normalize_pcd = normalize_pcd
        self.dt = 0.02  # Control timestep (50Hz)
        
        # Load normalization statistics
        if self.normalize:
            if action_stats_path is None:
                action_stats_path = self.hdf5_path.parent / "action_stats.json"
            if prop_stats_path is None:
                prop_stats_path = self.hdf5_path.parent / "prop_stats.json"
            if pcd_stats_path is None:
                pcd_stats_path = self.hdf5_path.parent / "pcd_stats.json"
            
            print(f"Loading normalization stats:")
            print(f"  - Action stats: {action_stats_path}")
            print(f"  - Prop stats: {prop_stats_path}")
            print(f"  - PCD stats: {pcd_stats_path}")
            
            with open(action_stats_path, 'r') as f:
                self.action_stats = json.load(f)
            with open(prop_stats_path, 'r') as f:
                self.prop_stats = json.load(f)
            with open(pcd_stats_path, 'r') as f:
                self.pcd_stats = json.load(f)
            
            # Convert to numpy arrays - extract from nested structure
            # JSON format: {"mobile_base": {"min": [...], "max": [...]}, "torso": {...}, "arms": {...}}
            action_min_list = []
            action_max_list = []
            
            # Mobile base (3D)
            action_min_list.extend(self.action_stats['mobile_base']['min'])
            action_max_list.extend(self.action_stats['mobile_base']['max'])
            
            # Torso (1D) - scalar value
            action_min_list.append(self.action_stats['torso']['min'])
            action_max_list.append(self.action_stats['torso']['max'])
            
            # Arms (12D)
            action_min_list.extend(self.action_stats['arms']['min'])
            action_max_list.extend(self.action_stats['arms']['max'])
            
            self.action_min = np.array(action_min_list)
            self.action_max = np.array(action_max_list)
            
            # Proprioception format: {"mobile_base_vel": {...}, "torso": {...}, ...}
            prop_min_list = []
            prop_max_list = []
            
            # Mobile base velocity (3D)
            prop_min_list.extend(self.prop_stats['mobile_base_vel']['min'])
            prop_max_list.extend(self.prop_stats['mobile_base_vel']['max'])
            
            # Torso (1D) - scalar value
            prop_min_list.append(self.prop_stats['torso']['min'])
            prop_max_list.append(self.prop_stats['torso']['max'])
            
            # Left arm (5D), Left gripper (1D), Right arm (5D), Right gripper (1D)
            for key in ['left_arm', 'left_gripper', 'right_arm', 'right_gripper']:
                val_min = self.prop_stats[key]['min']
                val_max = self.prop_stats[key]['max']
                if isinstance(val_min, list):
                    prop_min_list.extend(val_min)
                    prop_max_list.extend(val_max)
                else:
                    prop_min_list.append(val_min)
                    prop_max_list.append(val_max)
            
            self.prop_min = np.array(prop_min_list)
            self.prop_max = np.array(prop_max_list)
            
            # PCD XYZ normalization (3D: x, y, z)
            self.pcd_xyz_min = np.array(self.pcd_stats['xyz']['min'], dtype=np.float32)
            self.pcd_xyz_max = np.array(self.pcd_stats['xyz']['max'], dtype=np.float32)
        else:
            print("Normalization disabled")
        
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
        if self.normalize:
            print(f"  - Normalization: ENABLED (to [-1, 1])")
    
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
    
    def _normalize_to_range(self, value: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
        """
        Normalize value to [-1, 1] range
        
        Args:
            value: Input values (can be 1D or 2D)
            min_val: Minimum values per dimension
            max_val: Maximum values per dimension
        
        Returns:
            Normalized values in [-1, 1]
        """
        # Avoid division by zero
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1.0, range_val)
        
        # Normalize to [-1, 1]
        normalized = 2.0 * (value - min_val) / range_val - 1.0
        
        # Clip to ensure [-1, 1] range (in case of outliers)
        return np.clip(normalized, -1.0, 1.0)
    
    def _denormalize_from_range(self, value: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
        """
        Denormalize from [-1, 1] back to original range
        
        Args:
            value: Normalized values in [-1, 1]
            min_val: Minimum values per dimension
            max_val: Maximum values per dimension
        
        Returns:
            Original scale values
        """
        return min_val + (value + 1.0) * (max_val - min_val) / 2.0
    
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
        
        # Normalize XYZ to [-1, 1] range (only if not already preprocessed)
        if self.normalize_pcd:
            xyz = self._normalize_to_range(xyz, self.pcd_xyz_min, self.pcd_xyz_max)
        
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
            
            # Normalize XYZ to [-1, 1] range (only if not already preprocessed)
            if self.normalize_pcd:
                xyz = self._normalize_to_range(xyz, self.pcd_xyz_min, self.pcd_xyz_max)
            
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
        
        f = self._get_hdf5_file()
        demo_data = f['data'][demo_id]
        obs_indices = range(start_frame, start_frame + self.num_latest_obs)
        
        # Extract proprioception components
        prop_data = demo_data['obs']['proprioception'][obs_indices]
        gripper = demo_data['obs']['proprioception_grippers'][obs_indices]
        
        # Get positions from proprioception_floating_base (direct measurements, not accumulated)
        floating_base = demo_data['obs']['proprioception_floating_base'][obs_indices]
        
        # Calculate mobile base velocity from position changes
        # We need velocity that matches what the actions represent
        if start_frame > 0:
            # Get previous frame to compute velocity
            prev_frame_idx = start_frame - 1
            prev_floating_base = demo_data['obs']['proprioception_floating_base'][prev_frame_idx]
            
            # Compute velocity as (current_position - previous_position) / dt for each frame
            mobile_base_vel = np.zeros((len(obs_indices), 3), dtype=np.float32)
            for i, frame_idx in enumerate(obs_indices):
                if i == 0:
                    # First observation frame: use diff with previous frame
                    mobile_base_vel[i] = (floating_base[i, [0, 1, 3]] - prev_floating_base[[0, 1, 3]]) / self.dt
                else:
                    # Subsequent frames: use diff with previous observation frame
                    mobile_base_vel[i] = (floating_base[i, [0, 1, 3]] - floating_base[i-1, [0, 1, 3]]) / self.dt
        else:
            # First frame of demo: can't compute velocity from diff, approximate as zero or small value
            mobile_base_vel = np.zeros((len(obs_indices), 3), dtype=np.float32)
        
        # Get torso absolute position directly from proprioception_floating_base
        torso = floating_base[:, 2:3]  # Z position (pelvis_z)
        
        # Extract arm and gripper positions
        left_arm = np.concatenate([prop_data[:, 0:4], prop_data[:, 12:13]], axis=-1)
        left_gripper = gripper[:, 0:1]
        right_arm = np.concatenate([prop_data[:, 13:17], prop_data[:, 25:26]], axis=-1)
        right_gripper = gripper[:, 1:2]
        
        proprioception_concat = np.concatenate([
            mobile_base_vel, torso, left_arm, left_gripper, right_arm, right_gripper
        ], axis=-1)
        
        # Normalize proprioception
        if self.normalize:
            proprioception_concat = self._normalize_to_range(
                proprioception_concat, self.prop_min, self.prop_max
            )
        
        # Split normalized proprioception
        mobile_base_vel = proprioception_concat[:, 0:3]
        torso = proprioception_concat[:, 3:4]
        left_arm = proprioception_concat[:, 4:9]
        left_gripper = proprioception_concat[:, 9:10]
        right_arm = proprioception_concat[:, 10:15]
        right_gripper = proprioception_concat[:, 15:16]
        
        # Load and process actions (future actions for prediction)
        dt = 0.02
        action_indices = range(start_frame, start_frame + self.action_prediction_horizon)
        actions = demo_data['actions'][action_indices].copy()
        
        # Convert torso deltas to absolute positions
        # Start from the last observed torso position and accumulate deltas
        current_torso_pos = floating_base[-1, 2]  # Last observed torso position from proprioception_floating_base
        torso_deltas = actions[:, 2]  # Torso deltas from HDF5
        torso_absolute = current_torso_pos + np.cumsum(torso_deltas)  # Accumulate to get absolute positions
        
        # Convert mobile base deltas to velocity
        actions[:, [0, 1, 3]] /= dt  # Mobile base X, Y, RZ delta -> velocity
        # Replace torso deltas with absolute positions
        actions[:, 2] = torso_absolute
        # Arms and grippers are already absolute positions
        
        if self.normalize:
            actions = self._normalize_to_range(actions, self.action_min, self.action_max)
        
        # Split actions into components
        mobile_base_actions = np.concatenate([actions[:, 0:2], actions[:, 3:4]], axis=-1)
        torso_actions = actions[:, 2:3]  # Torso absolute position
        arms_actions = actions[:, 4:16]  # Arms and grippers (absolute positions)
        
        # Load point clouds from PCD files
        max_points_fixed = self.max_points_per_camera * len(self.cameras)
        pcd_xyz_padded = np.zeros((self.num_latest_obs, max_points_fixed, 3), dtype=np.float32)
        pcd_rgb_padded = np.zeros((self.num_latest_obs, max_points_fixed, 3), dtype=np.float32)
        
        for t, frame_idx in enumerate(obs_indices):
            xyz, rgb = self._load_multi_camera_pcd(demo_id, frame_idx)
            
            # Normalize PCD data
            if self.normalize:
                # Normalize XYZ to [-1, 1]
                xyz = self._normalize_to_range(xyz, self.pcd_xyz_min, self.pcd_xyz_max)
                # Normalize RGB from [0, 255] to [-1, 1]
                rgb = (rgb / 127.5) - 1.0
            
            n = min(len(xyz), max_points_fixed)
            pcd_xyz_padded[t, :n] = xyz[:n]
            pcd_rgb_padded[t, :n] = rgb[:n]
        
        # Construct output dictionary
        return {
            "odom": {
                "base_velocity": mobile_base_vel.astype(np.float32),
            },
            "qpos": {
                "torso": torso.astype(np.float32),
                "left_arm": left_arm.astype(np.float32),
                "left_gripper": left_gripper.astype(np.float32),
                "right_arm": right_arm.astype(np.float32),
                "right_gripper": right_gripper.astype(np.float32),
            },
            "pointcloud": {
                "xyz": pcd_xyz_padded,
                "rgb": pcd_rgb_padded,
            },
            "action_chunks": {
                "mobile_base": np.tile(mobile_base_actions[None, :, :], (self.num_latest_obs, 1, 1)).astype(np.float32),
                "torso": np.tile(torso_actions[None, :, :], (self.num_latest_obs, 1, 1)).astype(np.float32),
                "arms": np.tile(arms_actions[None, :, :], (self.num_latest_obs, 1, 1)).astype(np.float32),
            },
            "pad_mask": np.ones((self.num_latest_obs, self.action_prediction_horizon), dtype=np.float32),
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
        normalize: bool = True,
        action_stats_path: Optional[str] = None,
        prop_stats_path: Optional[str] = None,
        pcd_stats_path: Optional[str] = None,
        normalize_pcd: bool = True,
        subsample_points: bool = True,
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
        self.normalize = normalize
        self.action_stats_path = action_stats_path
        self.prop_stats_path = prop_stats_path
        self.pcd_stats_path = pcd_stats_path
        self.normalize_pcd = normalize_pcd
        self.subsample_points = subsample_points
        self.normalize = normalize
        self.action_stats_path = action_stats_path
        self.prop_stats_path = prop_stats_path
        
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
            subsample_points=self.subsample_points,  # Use config value
            normalize=self.normalize,
            action_stats_path=self.action_stats_path,
            prop_stats_path=self.prop_stats_path,
            pcd_stats_path=self.pcd_stats_path,
            normalize_pcd=self.normalize_pcd,  # Separate flag for PCD normalization
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
            normalize=self.normalize,
            action_stats_path=self.action_stats_path,
            prop_stats_path=self.prop_stats_path,
            pcd_stats_path=self.pcd_stats_path,
            normalize_pcd=self.normalize_pcd,  # Separate flag for PCD normalization
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
