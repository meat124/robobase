"""
PCD Dataset for BRS Policy Training

Designed for the following HDF5 structure (BigYM SaucepanToHob):
    demo_N/
        actions: (T, 16) float32
        proprioception: (T+1, 60) float32  # [qpos(30) + qvel(30)]
        proprioception_floating_base: (T+1, 4) float32  # [x, y, z, rz] absolute
        proprioception_floating_base_actions: (T+1, 4) float32  # [dx, dy, dz, drz] delta
        proprioception_grippers: (T+1, 2) float32  # [left, right]
        rgb_head: (T+1, 3, 224, 224) uint8
        depth_head: (T+1, 224, 224) float32

PCD files are stored separately:
    pcd/
        demo_XXX_pcd.npy  # (T+1, 4096, 3) float32

BigYM 16D Action Structure:
    - action[0:4]   = floating_base delta [dx, dy, dz, drz] - DELTA mode
    - action[4:9]   = left_arm (5D) - ABSOLUTE position
    - action[9:14]  = right_arm (5D) - ABSOLUTE position
    - action[14:16] = grippers (2D) - ABSOLUTE [0, 1]

BRS Policy expects 16D actions in 3-part autoregressive structure:
    - mobile_base: 3D (x_vel, y_vel, rz_vel) - from action[0,1,3]/dt
    - torso: 1D (z position delta) - from action[2]
    - arms: 12D (left_arm 5 + left_gripper 1 + right_arm 5 + right_gripper 1)

Proprioception qpos mapping (from README):
    qpos[1,2,3,4]   = left_arm (shoulder, roll, yaw, elbow)
    qpos[13]        = left_wrist
    qpos[14,15,16,17] = right_arm
    qpos[26]        = right_wrist
    qpos[12]        = left_gripper
    qpos[21]        = right_gripper
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import json


# qpos indices for BigYM H1 robot (verified from robot._joints order)
# qpos[0-3]: left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow
# qpos[4-11]: left_gripper joints (8)
# qpos[12]: left_wrist
# qpos[13-16]: right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow
# qpos[17-24]: right_gripper joints (8)
# qpos[25]: right_wrist
# qpos[26-29]: pelvis_x, pelvis_y, pelvis_z, pelvis_rz
QPOS_LEFT_ARM = [0, 1, 2, 3, 12]       # 5D: shoulder_pitch, roll, yaw, elbow, wrist
QPOS_RIGHT_ARM = [13, 14, 15, 16, 25]  # 5D: shoulder_pitch, roll, yaw, elbow, wrist
QPOS_LEFT_GRIPPER = 4   # left gripper driver joint
QPOS_RIGHT_GRIPPER = 17  # right gripper driver joint


class PCDBRSDataset(Dataset):
    """
    Dataset for BRS Policy training with separate PCD files.
    
    Handles BigYM SaucepanToHob HDF5 format with 16D actions.
    
    Supports preloading data into RAM for faster training:
    - preload_hdf5: Load all HDF5 data (proprioception, actions) into memory
    - preload_pcd: Load all PCD files into memory
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
        """
        Args:
            hdf5_path: Path to HDF5 file (demos.hdf5)
            pcd_root: Root directory containing PCD .npz files
            demo_ids: List of demo IDs to use (e.g., ['demo_0', 'demo_1', ...])
            cameras: Camera names for PCD loading (default: ['head'])
            num_latest_obs: Temporal window size for observations
            action_prediction_horizon: Number of future actions to predict
            max_points: Maximum total points (all cameras combined)
            subsample_points: Whether to randomly subsample points
            action_stats_path: Path to action statistics JSON
            prop_stats_path: Path to proprioception statistics JSON  
            pcd_stats_path: Path to PCD statistics JSON
            normalize: Whether to normalize actions/proprioception to [-1, 1]
            normalize_pcd: Whether to normalize PCD XYZ coordinates to [-1, 1]
            preload_hdf5: Preload all HDF5 data into RAM (faster training, ~100MB for 30 demos)
            preload_pcd: Preload all PCD files into RAM (faster training, ~3GB for 30 demos)
        """
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
        self.dt = 0.02  # 50Hz control
        
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
        print(f"  - Window size: {num_latest_obs}")
        print(f"  - Action horizon: {action_prediction_horizon}")
        print(f"  - Normalize: {normalize}")
    
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
        # action_stats.json structure (new format):
        #   mobile_base: {min: [3], max: [3], ...}  # [dx, dy, drz]
        #   torso: {min: scalar, max: scalar, ...}
        #   arms: {min: [12], max: [12], ...}       # [left_arm(5) + left_grip(1) + right_arm(5) + right_grip(1)]
        #   full: {min: [16], max: [16], ...}       # Complete 16D action
        # BRS structure: mobile_base(3) + torso(1) + arms(12) = 16D
        if Path(action_stats_path).exists():
            with open(action_stats_path, 'r') as f:
                action_stats = json.load(f)
            
            # Build 16D action bounds for BRS: mobile_base(3) + torso(1) + arms(12)
            action_min = []
            action_max = []
            
            # Check for new format with 'full' key
            if 'full' in action_stats:
                # New format: use 'full' directly (already 16D BRS format)
                self.action_min = np.array(action_stats['full']['min'], dtype=np.float32)
                self.action_max = np.array(action_stats['full']['max'], dtype=np.float32)
                print(f"  Loaded action stats (full): shape={self.action_min.shape}")
            
            # Check if stats have flat format or grouped format
            elif 'min' in action_stats and isinstance(action_stats['min'], list):
                # Flat format: directly use 16D stats
                # BigYM 16D: [fb(4), left_arm(5), right_arm(5), grippers(2)]
                # BRS 16D: [mobile_base(3), torso(1), arms(12)]
                fb_min = action_stats['min'][0:4]
                fb_max = action_stats['max'][0:4]
                arms_grippers_min = action_stats['min'][4:16]
                arms_grippers_max = action_stats['max'][4:16]
                
                # mobile_base: fb[0,1,3] (x, y, rz)
                action_min.extend([fb_min[0], fb_min[1], fb_min[3]])
                action_max.extend([fb_max[0], fb_max[1], fb_max[3]])
                
                # torso: fb[2] (z)
                action_min.append(fb_min[2])
                action_max.append(fb_max[2])
                
                # arms: left_arm(5) + left_grip(1) + right_arm(5) + right_grip(1) = 12D
                # From BigYM: [left_arm(5), right_arm(5), grippers(2)]
                # Need: [left_arm(5), left_grip(1), right_arm(5), right_grip(1)]
                action_min.extend(arms_grippers_min[0:5])   # left_arm
                action_min.append(arms_grippers_min[10])    # left_gripper
                action_min.extend(arms_grippers_min[5:10])  # right_arm
                action_min.append(arms_grippers_min[11])    # right_gripper
                
                action_max.extend(arms_grippers_max[0:5])
                action_max.append(arms_grippers_max[10])
                action_max.extend(arms_grippers_max[5:10])
                action_max.append(arms_grippers_max[11])
                
                self.action_min = np.array(action_min, dtype=np.float32)
                self.action_max = np.array(action_max, dtype=np.float32)
                print(f"  Loaded action stats (flat): shape={self.action_min.shape}")
                
            elif 'floating_base' in action_stats:
                # Grouped format
                fb = action_stats['floating_base']
                arms = action_stats['arms']
                
                # mobile_base: [x, y, rz] from floating_base[0,1,3]
                action_min.extend([fb['min'][0], fb['min'][1], fb['min'][3]])
                action_max.extend([fb['max'][0], fb['max'][1], fb['max'][3]])
                
                # torso: z from floating_base[2]
                action_min.append(fb['min'][2])
                action_max.append(fb['max'][2])
                
                # arms (10D from stats) -> 12D for BRS (add grippers)
                # arms stats: [left_arm(5), right_arm(5)] - grippers might be missing
                if len(arms['min']) == 10:
                    # No grippers in arms stats, assume [0, 1] range
                    action_min.extend(arms['min'][0:5])   # left_arm
                    action_min.append(0.0)                 # left_gripper
                    action_min.extend(arms['min'][5:10])  # right_arm
                    action_min.append(0.0)                 # right_gripper
                    
                    action_max.extend(arms['max'][0:5])
                    action_max.append(1.0)
                    action_max.extend(arms['max'][5:10])
                    action_max.append(1.0)
                elif len(arms['min']) == 12:
                    action_min.extend(arms['min'])
                    action_max.extend(arms['max'])
                
                self.action_min = np.array(action_min, dtype=np.float32)
                self.action_max = np.array(action_max, dtype=np.float32)
                print(f"  Loaded action stats (grouped): shape={self.action_min.shape}")
            else:
                print(f"  Warning: Unknown action_stats format, keys: {list(action_stats.keys())}")
                self.action_min = self.action_max = None
        else:
            print(f"  Warning: Action stats not found at {action_stats_path}")
            self.action_min = self.action_max = None
        
        # Load proprioception stats
        # prop_stats.json structure:
        #   mobile_base_vel: {min: [3], max: [3], ...}
        #   torso: {min: scalar, max: scalar, ...}
        #   left_arm: {min: [5], max: [5], ...}
        #   left_gripper: {min: scalar, max: scalar, ...}
        #   right_arm: {min: [5], max: [5], ...}
        #   right_gripper: {min: scalar, max: scalar, ...}
        if Path(prop_stats_path).exists():
            with open(prop_stats_path, 'r') as f:
                prop_stats = json.load(f)
            
            # Build 16D prop bounds: mobile_base_vel(3) + torso(1) + arms(12)
            prop_min = []
            prop_max = []
            
            # Mobile base velocity (3D)
            prop_min.extend(prop_stats['mobile_base_vel']['min'])
            prop_max.extend(prop_stats['mobile_base_vel']['max'])
            
            # Torso (1D)
            prop_min.append(prop_stats['torso']['min'])
            prop_max.append(prop_stats['torso']['max'])
            
            # Arms: left_arm(5) + left_gripper(1) + right_arm(5) + right_gripper(1)
            for key in ['left_arm', 'left_gripper', 'right_arm', 'right_gripper']:
                val_min = prop_stats[key]['min']
                val_max = prop_stats[key]['max']
                if isinstance(val_min, list):
                    prop_min.extend(val_min)
                    prop_max.extend(val_max)
                else:
                    prop_min.append(val_min)
                    prop_max.append(val_max)
            
            self.prop_min = np.array(prop_min, dtype=np.float32)
            self.prop_max = np.array(prop_max, dtype=np.float32)
            print(f"  Loaded prop stats: shape={self.prop_min.shape}")
        else:
            print(f"  Warning: Prop stats not found at {prop_stats_path}")
            self.prop_min = self.prop_max = None
        
        # Load PCD stats
        if Path(pcd_stats_path).exists():
            with open(pcd_stats_path, 'r') as f:
                pcd_stats = json.load(f)
            self.pcd_xyz_min = np.array(pcd_stats['xyz']['min'], dtype=np.float32)
            self.pcd_xyz_max = np.array(pcd_stats['xyz']['max'], dtype=np.float32)
            print(f"  Loaded PCD stats: min={self.pcd_xyz_min}, max={self.pcd_xyz_max}")
        else:
            print(f"  Warning: PCD stats not found at {pcd_stats_path}")
            self.pcd_xyz_min = self.pcd_xyz_max = None
    
    def _build_sample_index(self) -> List[Tuple[str, int, int]]:
        """Build index of valid samples: (demo_id, start_frame, num_frames)."""
        samples = []
        
        with h5py.File(self.hdf5_path, 'r') as f:
            for demo_id in self.demo_ids:
                if demo_id not in f:
                    print(f"  Warning: {demo_id} not found in HDF5")
                    continue
                    
                demo_data = f[demo_id]
                
                # Get number of action frames
                num_frames = demo_data['actions'].shape[0]
                
                # Verify action dimension is 16
                action_dim = demo_data['actions'].shape[1]
                if action_dim != 16:
                    print(f"  Warning: {demo_id} has {action_dim}D actions, expected 16D")
                
                # Check PCD availability
                if self.pcd_root:
                    demo_idx = int(demo_id.split('_')[1])
                    # Support both new format (demo_XXX_pcd.npy) and old format (demo_N/)
                    npy_path = self.pcd_root / f"demo_{demo_idx:03d}_pcd.npy"
                    pcd_dir = self.pcd_root / demo_id
                    if not npy_path.exists() and not pcd_dir.exists():
                        print(f"  Warning: PCD not found for {demo_id}")
                        continue
                
                # Create samples with sliding window
                min_frames = self.num_latest_obs + self.action_prediction_horizon
                for start_idx in range(max(0, num_frames - min_frames + 1)):
                    samples.append((demo_id, start_idx, num_frames))
        
        return samples
    
    def _preload_hdf5_data(self):
        """Preload all HDF5 data into RAM for faster training."""
        print("  Preloading HDF5 data into RAM...")
        self._preloaded_hdf5 = {}
        
        with h5py.File(self.hdf5_path, 'r') as f:
            for demo_id in self.demo_ids:
                if demo_id not in f:
                    continue
                
                demo_data = f[demo_id]
                self._preloaded_hdf5[demo_id] = {
                    'actions': demo_data['actions'][:],  # (T, 16)
                    'proprioception': demo_data['proprioception'][:],  # (T+1, 60)
                }
                
                # Load optional datasets
                if 'proprioception_floating_base' in demo_data:
                    self._preloaded_hdf5[demo_id]['proprioception_floating_base'] = \
                        demo_data['proprioception_floating_base'][:]  # (T+1, 4)
                
                if 'proprioception_grippers' in demo_data:
                    self._preloaded_hdf5[demo_id]['proprioception_grippers'] = \
                        demo_data['proprioception_grippers'][:]  # (T+1, 2)
        
        # Calculate memory usage
        total_bytes = 0
        for demo_id, data in self._preloaded_hdf5.items():
            for key, arr in data.items():
                total_bytes += arr.nbytes
        
        print(f"  Preloaded HDF5: {len(self._preloaded_hdf5)} demos, {total_bytes / 1024 / 1024:.1f} MB")
    
    def _preload_pcd_data(self):
        """Preload all PCD data into RAM for faster training."""
        if self.pcd_root is None:
            print("  Warning: pcd_root is None, skipping PCD preload")
            return
        
        print("  Preloading PCD data into RAM...")
        
        total_bytes = 0
        loaded_count = 0
        
        for demo_id in self.demo_ids:
            demo_idx = int(demo_id.split('_')[1])
            npy_path = self.pcd_root / f"demo_{demo_idx:03d}_pcd.npy"
            
            if npy_path.exists():
                pcd_data = np.load(npy_path)  # (T+1, N, 3)
                self._pcd_cache[demo_id] = pcd_data
                total_bytes += pcd_data.nbytes
                loaded_count += 1
            else:
                # Old format: load all .npz files for this demo
                # This is slower but we preload all frames
                pcd_dir = self.pcd_root / demo_id
                if pcd_dir.exists():
                    frames_data = {}
                    for camera in self.cameras:
                        camera_dir = pcd_dir / camera
                        if camera_dir.exists():
                            for npz_file in sorted(camera_dir.glob("frame_*.npz")):
                                frame_idx = int(npz_file.stem.split('_')[1])
                                data = np.load(npz_file)
                                if frame_idx not in frames_data:
                                    frames_data[frame_idx] = {'xyz': [], 'rgb': []}
                                frames_data[frame_idx]['xyz'].append(data['xyz'])
                                frames_data[frame_idx]['rgb'].append(data['rgb'])
                                total_bytes += data['xyz'].nbytes + data['rgb'].nbytes
                    
                    if frames_data:
                        self._pcd_cache[f"{demo_id}_frames"] = frames_data
                        loaded_count += 1
        
        print(f"  Preloaded PCD: {loaded_count} demos, {total_bytes / 1024 / 1024:.1f} MB")
    
    def _get_hdf5_file(self):
        """Get HDF5 file handle (per-worker for multi-process DataLoader)."""
        import torch.utils.data
        
        worker_info = torch.utils.data.get_worker_info()
        current_worker = worker_info.id if worker_info else -1
        
        # Open new handle if needed (different worker or first access)
        if not hasattr(self, '_worker_id'):
            self._worker_id = None
            
        if self._hdf5_file is None or self._worker_id != current_worker:
            if self._hdf5_file is not None:
                self._hdf5_file.close()
            self._hdf5_file = h5py.File(
                self.hdf5_path, 'r',
                rdcc_nbytes=256*1024*1024,  # 256MB chunk cache
                rdcc_nslots=10007,  # Prime number for hash table
            )
            self._worker_id = current_worker
        return self._hdf5_file
    
    def _normalize_to_range(self, value: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
        """Normalize value to [-1, 1] range."""
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1.0, range_val)
        normalized = 2.0 * (value - min_val) / range_val - 1.0
        return np.clip(normalized, -1.0, 1.0)
    
    def _load_pcd_frame(self, demo_id: str, frame_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load PCD from files.
        
        Supports two formats:
        1. New format: pcd/demo_XXX_pcd.npy - (T+1, N, 3) array
        2. Old format: pcd_root/demo_N/camera/frame_XXXXX.npz
        """
        all_xyz = []
        all_rgb = []
        
        # Try new format first: single .npy file per demo
        demo_idx = int(demo_id.split('_')[1])
        npy_path = self.pcd_root / f"demo_{demo_idx:03d}_pcd.npy"
        
        if npy_path.exists():
            # New format: load from single .npy file
            if not hasattr(self, '_pcd_cache'):
                self._pcd_cache = {}
            
            if demo_id not in self._pcd_cache:
                self._pcd_cache[demo_id] = np.load(npy_path)  # (T+1, N, 3)
            
            pcd_data = self._pcd_cache[demo_id]
            if frame_idx < len(pcd_data):
                xyz = pcd_data[frame_idx].astype(np.float32)  # (N, 3)
                # No RGB in new format, use zeros
                rgb = np.zeros_like(xyz)
                all_xyz.append(xyz)
                all_rgb.append(rgb)
        else:
            # Old format: load from per-frame .npz files
            for camera in self.cameras:
                npz_path = self.pcd_root / demo_id / camera / f"frame_{frame_idx:05d}.npz"
            
                if npz_path.exists():
                    data = np.load(npz_path)
                    xyz = data['xyz'].astype(np.float32)
                    rgb = data['rgb'].astype(np.float32)
                    
                    all_xyz.append(xyz)
                    all_rgb.append(rgb)
        
        if len(all_xyz) == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.float32)
        
        xyz = np.concatenate(all_xyz, axis=0)
        rgb = np.concatenate(all_rgb, axis=0)
        
        # Subsample if needed
        if self.subsample_points and len(xyz) > self.max_points:
            indices = np.random.choice(len(xyz), self.max_points, replace=False)
            xyz = xyz[indices]
            rgb = rgb[indices]
        
        # Normalize XYZ to [-1, 1]
        if self.normalize_pcd and self.pcd_xyz_min is not None:
            xyz = self._normalize_to_range(xyz, self.pcd_xyz_min, self.pcd_xyz_max)
        
        return xyz, rgb
    
    def _extract_proprioception(self, demo_data, obs_indices, actions) -> Dict[str, np.ndarray]:
        """
        Extract proprioception components from HDF5.
        
        New HDF5 format:
            proprioception: (T+1, 60) = qpos(30) + qvel(30)
            proprioception_floating_base: (T+1, 4) = [x, y, z, rz] absolute
            proprioception_floating_base_actions: (T+1, 4) = [dx, dy, dz, drz] cumulative actions
            proprioception_grippers: (T+1, 2) = [left, right]
        
        Returns dict with BRS format (modified to match ACT):
            - mobile_base_vel: (T, 3) - floating_base [x, y, rz] ABSOLUTE position (NOT velocity!)
            - torso: (T, 1) - z position from proprioception_floating_base
            - left_arm: (T, 5) - from qpos[0,1,2,3,12]
            - left_gripper: (T, 1) - from proprioception_grippers[0]
            - right_arm: (T, 5) - from qpos[13,14,15,16,25]
            - right_gripper: (T, 1) - from proprioception_grippers[1]
        
        Note: Despite the name 'mobile_base_vel', we use ABSOLUTE position (not velocity)
              to be consistent with ACT policy and prop_stats.json.
        """
        # Handle both dict (preloaded) and h5py dataset formats
        if isinstance(demo_data, dict):
            prop_full = demo_data['proprioception']
            fb_data = demo_data.get('proprioception_floating_base', None)
            gripper_data = demo_data.get('proprioception_grippers', None)
        else:
            prop_full = demo_data['proprioception'][:]
            fb_data = demo_data['proprioception_floating_base'][:] if 'proprioception_floating_base' in demo_data else None
            gripper_data = demo_data['proprioception_grippers'][:] if 'proprioception_grippers' in demo_data else None
        
        prop = prop_full[obs_indices]  # (T, 60)
        T = len(obs_indices)
        
        # Extract arm joints from qpos (first 30 dims of proprioception)
        # Left arm: qpos[0,1,2,3,12] (shoulder_pitch, roll, yaw, elbow, wrist)
        left_arm = np.column_stack([
            prop[:, 0], prop[:, 1], prop[:, 2], prop[:, 3], prop[:, 12]
        ])  # (T, 5)
        
        # Right arm: qpos[13,14,15,16,25] (shoulder_pitch, roll, yaw, elbow, wrist)
        right_arm = np.column_stack([
            prop[:, 13], prop[:, 14], prop[:, 15], prop[:, 16], prop[:, 25]
        ])  # (T, 5)
        
        # Grippers: use proprioception_grippers directly if available, else from qpos
        if gripper_data is not None:
            grippers = gripper_data[obs_indices]  # (T, 2)
            left_gripper = grippers[:, 0:1]   # (T, 1)
            right_gripper = grippers[:, 1:2]  # (T, 1)
        else:
            # Fallback to qpos indices (gripper driver joints)
            left_gripper = prop[:, 4:5]    # (T, 1) - left gripper driver
            right_gripper = prop[:, 17:18]  # (T, 1) - right gripper driver
        
        # Mobile base: use ABSOLUTE position from proprioception_floating_base
        # This is consistent with ACT policy which uses floating_base position directly
        # Despite the variable name 'mobile_base_vel', this is position, not velocity
        if fb_data is not None:
            fb_pos = fb_data[obs_indices]  # (T, 4)
            # Extract [x, y, rz] from [x, y, z, rz]
            mobile_base = fb_pos[:, [0, 1, 3]]  # (T, 3) - [x, y, rz] absolute position
        else:
            # Fallback: use zeros (should not happen with proper HDF5)
            mobile_base = np.zeros((T, 3), dtype=np.float32)
        
        # Torso (z position): use proprioception_floating_base if available
        if fb_data is not None:
            fb_pos = fb_data[obs_indices]  # (T, 4)
            torso = fb_pos[:, 2:3]  # z position (T, 1)
        else:
            # Fallback: integrate action[2] deltas
            torso = np.zeros((T, 1), dtype=np.float32)
            init_z = 1.0
            for i, idx in enumerate(obs_indices):
                if idx > 0:
                    cumsum_z = init_z + actions[:idx, 2].sum()
                    torso[i, 0] = cumsum_z
                else:
                    torso[i, 0] = init_z
        
        return {
            'mobile_base_vel': mobile_base.astype(np.float32),  # (T, 3) - actually position [x, y, rz]
            'torso': torso.astype(np.float32),  # (T, 1)
            'left_arm': left_arm.astype(np.float32),  # (T, 5)
            'left_gripper': left_gripper.astype(np.float32),  # (T, 1)
            'right_arm': right_arm.astype(np.float32),  # (T, 5)
            'right_gripper': right_gripper.astype(np.float32),  # (T, 1)
        }
    
    def _process_actions(self, actions_raw: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Process 16D BigYM actions to BRS 16D format.
        
        BigYM 16D: [fb_x, fb_y, fb_z, fb_rz, left_arm(5), right_arm(5), grippers(2)]
        BRS 16D:   [mobile_base(3), torso(1), arms(12)]
        
        Args:
            actions_raw: (horizon, 16) raw actions from HDF5
            
        Returns dict with:
            - mobile_base: (horizon, 3) - delta [dx, dy, drz] (same as HDF5, NOT velocity)
            - torso: (horizon, 1) - z delta
            - arms: (horizon, 12) - [left_arm(5), left_grip(1), right_arm(5), right_grip(1)]
        
        Note: mobile_base is kept as DELTA (not velocity) to match action_stats.json
              and to be consistent with ACT policy processing.
        """
        horizon = actions_raw.shape[0]
        
        # Extract floating base: [dx, dy, dz, drz]
        fb_x = actions_raw[:, 0]   # x delta
        fb_y = actions_raw[:, 1]   # y delta
        fb_z = actions_raw[:, 2]   # z delta
        fb_rz = actions_raw[:, 3]  # rz delta
        
        # Mobile base: [dx, dy, drz] - keep as DELTA (NOT velocity!)
        # This matches action_stats.json which has delta stats, not velocity stats
        mobile_base = np.column_stack([fb_x, fb_y, fb_rz])  # (horizon, 3)
        
        # Torso: z delta
        torso = fb_z.reshape(-1, 1)  # (horizon, 1)
        
        # Arms from BigYM: [left_arm(5), right_arm(5), grippers(2)]
        # indices: 4:9 = left_arm, 9:14 = right_arm, 14:16 = grippers
        left_arm = actions_raw[:, 4:9]    # (horizon, 5)
        right_arm = actions_raw[:, 9:14]  # (horizon, 5)
        left_gripper = actions_raw[:, 14:15]   # (horizon, 1)
        right_gripper = actions_raw[:, 15:16]  # (horizon, 1)
        
        # Combine to BRS format: [left_arm(5), left_grip(1), right_arm(5), right_grip(1)]
        arms = np.concatenate([
            left_arm, left_gripper, right_arm, right_gripper
        ], axis=-1)  # (horizon, 12)
        
        return {
            'mobile_base': mobile_base.astype(np.float32),  # (horizon, 3)
            'torso': torso.astype(np.float32),              # (horizon, 1)
            'arms': arms.astype(np.float32),                # (horizon, 12)
        }
    
    def __len__(self):
        return len(self.samples)
    
    def _get_demo_data(self, demo_id: str):
        """Get demo data from preloaded cache or HDF5 file."""
        if self._preloaded_hdf5 is not None and demo_id in self._preloaded_hdf5:
            # Return preloaded data as dict-like object
            return self._preloaded_hdf5[demo_id]
        else:
            # Fallback to HDF5 file access
            f = self._get_hdf5_file()
            return f[demo_id]
    
    def __getitem__(self, idx):
        demo_id, start_frame, num_frames = self.samples[idx]
        
        demo_data = self._get_demo_data(demo_id)
        
        # Load all actions for this demo (needed for proprioception computation)
        if isinstance(demo_data, dict):
            # Preloaded data
            all_actions = demo_data['actions']  # Already numpy array
        else:
            # HDF5 dataset
            all_actions = demo_data['actions'][:]  # (N, 16)
        
        # Observation indices
        obs_indices = list(range(start_frame, start_frame + self.num_latest_obs))
        
        # Action indices (future actions for prediction)
        action_start = start_frame + self.num_latest_obs - 1
        action_end = min(action_start + self.action_prediction_horizon, num_frames)
        action_indices = list(range(action_start, action_end))
        
        # Pad actions if needed
        n_actions = len(action_indices)
        pad_length = self.action_prediction_horizon - n_actions
        
        # Get raw actions for the prediction horizon
        actions_raw = all_actions[action_indices]  # (n_actions, 16)
        
        # Process actions to BRS format
        action_data = self._process_actions(actions_raw)
        
        # Pad actions if needed
        if pad_length > 0:
            for key in action_data:
                pad_shape = (pad_length,) + action_data[key].shape[1:]
                padding = np.zeros(pad_shape, dtype=np.float32)
                action_data[key] = np.concatenate([action_data[key], padding], axis=0)
        
        # Extract proprioception
        prop_data = self._extract_proprioception(demo_data, obs_indices, all_actions)
        
        # Normalize proprioception
        if self.normalize and self.prop_min is not None:
            prop_concat = np.concatenate([
                prop_data['mobile_base_vel'],
                prop_data['torso'],
                prop_data['left_arm'],
                prop_data['left_gripper'],
                prop_data['right_arm'],
                prop_data['right_gripper'],
            ], axis=-1)  # (T, 16)
            
            prop_concat = self._normalize_to_range(prop_concat, self.prop_min, self.prop_max)
            
            # Split back
            prop_data['mobile_base_vel'] = prop_concat[:, 0:3]
            prop_data['torso'] = prop_concat[:, 3:4]
            prop_data['left_arm'] = prop_concat[:, 4:9]
            prop_data['left_gripper'] = prop_concat[:, 9:10]
            prop_data['right_arm'] = prop_concat[:, 10:15]
            prop_data['right_gripper'] = prop_concat[:, 15:16]
        
        # Normalize actions
        if self.normalize and self.action_min is not None:
            action_concat = np.concatenate([
                action_data['mobile_base'],
                action_data['torso'],
                action_data['arms'],
            ], axis=-1)  # (horizon, 16)
            
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
        
        # Create padding mask
        pad_mask = np.ones((self.num_latest_obs, self.action_prediction_horizon), dtype=np.float32)
        if pad_length > 0:
            pad_mask[:, -pad_length:] = 0.0
        
        # Tile actions for each observation timestep (BRS format)
        action_chunks = {
            'mobile_base': np.tile(action_data['mobile_base'][None, :, :], (self.num_latest_obs, 1, 1)),
            'torso': np.tile(action_data['torso'][None, :, :], (self.num_latest_obs, 1, 1)),
            'arms': np.tile(action_data['arms'][None, :, :], (self.num_latest_obs, 1, 1)),
        }
        
        return {
            'odom': {
                'base_velocity': prop_data['mobile_base_vel'],  # (T, 3)
            },
            'qpos': {
                'torso': prop_data['torso'],  # (T, 1)
                'left_arm': prop_data['left_arm'],  # (T, 5)
                'left_gripper': prop_data['left_gripper'],  # (T, 1)
                'right_arm': prop_data['right_arm'],  # (T, 5)
                'right_gripper': prop_data['right_gripper'],  # (T, 1)
            },
            'pointcloud': {
                'xyz': pcd_xyz,  # (T, max_points, 3)
                'rgb': pcd_rgb,  # (T, max_points, 3)
            },
            'action_chunks': action_chunks,
            'pad_mask': pad_mask,  # (T, horizon)
        }


def pcd_brs_collate_fn(batch):
    """Collate function for BRS dataset with N_chunks dimension."""
    
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
    
    # Add N_chunks dimension: (B, ...) -> (1, B, ...)
    return {
        "odom": {
            "base_velocity": odom_base_velocity.unsqueeze(0),
        },
        "qpos": {
            "torso": qpos_torso.unsqueeze(0),
            "left_arm": qpos_left_arm.unsqueeze(0),
            "left_gripper": qpos_left_gripper.unsqueeze(0),
            "right_arm": qpos_right_arm.unsqueeze(0),
            "right_gripper": qpos_right_gripper.unsqueeze(0),
        },
        "pointcloud": {
            "xyz": pcd_xyz.unsqueeze(0),
            "rgb": pcd_rgb.unsqueeze(0),
        },
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
        max_points: int = 4096,
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
        # Ignored parameters for compatibility
        max_points_per_camera: Optional[int] = None,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.pcd_root = pcd_root
        self.cameras = cameras
        self.num_latest_obs = num_latest_obs
        self.action_prediction_horizon = action_prediction_horizon
        self.max_points = max_points_per_camera if max_points_per_camera else max_points
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
        
        # Auto-discover demo_ids
        if demo_ids is None:
            self.demo_ids = self._discover_demo_ids()
        else:
            self.demo_ids = demo_ids
    
    def _discover_demo_ids(self) -> List[str]:
        """Auto-discover demo IDs from HDF5 file."""
        with h5py.File(self.hdf5_path, 'r') as f:
            demo_ids = sorted([k for k in f.keys() if k.startswith('demo_')])
        print(f"Auto-discovered {len(demo_ids)} demos")
        return demo_ids
    
    def setup(self, stage: Optional[str] = None):
        np.random.seed(self.seed or 42)
        demo_ids_shuffled = np.random.permutation(self.demo_ids).tolist()
        
        num_val = max(1, int(len(demo_ids_shuffled) * self.val_split_ratio))
        val_demo_ids = demo_ids_shuffled[:num_val]
        train_demo_ids = demo_ids_shuffled[num_val:]
        
        print(f"\nDataset split:")
        print(f"  Train demos ({len(train_demo_ids)}): {train_demo_ids[:5]}...")
        print(f"  Val demos ({len(val_demo_ids)}): {val_demo_ids}")
        
        common_kwargs = {
            'hdf5_path': self.hdf5_path,
            'pcd_root': self.pcd_root,
            'cameras': self.cameras,
            'num_latest_obs': self.num_latest_obs,
            'action_prediction_horizon': self.action_prediction_horizon,
            'max_points': self.max_points,
            'normalize': self.normalize,
            'action_stats_path': self.action_stats_path,
            'prop_stats_path': self.prop_stats_path,
            'pcd_stats_path': self.pcd_stats_path,
            'normalize_pcd': self.normalize_pcd,
            'preload_hdf5': self.preload_hdf5,
            'preload_pcd': self.preload_pcd,
        }
        
        self.train_dataset = PCDBRSDataset(
            demo_ids=train_demo_ids,
            subsample_points=self.subsample_points,
            **common_kwargs
        )
        
        self.val_dataset = PCDBRSDataset(
            demo_ids=val_demo_ids,
            subsample_points=False,  # No random subsampling for validation
            **common_kwargs
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
            collate_fn=pcd_brs_collate_fn,
            prefetch_factor=self.prefetch_factor if self.dataloader_num_workers > 0 else None,
            persistent_workers=self.persistent_workers and self.dataloader_num_workers > 0,
            pin_memory=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
            collate_fn=pcd_brs_collate_fn,
            prefetch_factor=self.prefetch_factor if self.dataloader_num_workers > 0 else None,
            persistent_workers=self.persistent_workers and self.dataloader_num_workers > 0,
            pin_memory=True,
        )
