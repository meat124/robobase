"""
ACT HDF5 Dataset for SaucepanToHob demonstrations (Updated for new HDF5 format).

Loads data directly from demos.hdf5 for ACT policy training.

New HDF5 Structure (from convert_to_hdf5_full_v2.py):
    demo_N/
        actions: (T, 16) float32
        proprioception: (T+1, 60) float32  # qpos(30) + qvel(30)
        proprioception_floating_base: (T+1, 4) float32  # [x, y, z, rz] absolute
        proprioception_floating_base_actions: (T+1, 4) float32  # [dx, dy, dz, drz] delta
        proprioception_grippers: (T+1, 2) float32  # [left, right]
        rgb_head: (T+1, 3, 224, 224) uint8
        rgb_left_wrist: (T+1, 3, 224, 224) uint8
        rgb_right_wrist: (T+1, 3, 224, 224) uint8
        depth_head: (T+1, 224, 224) float32
        depth_left_wrist: (T+1, 224, 224) float32
        depth_right_wrist: (T+1, 224, 224) float32

Action Space (16D):
    [0:4]   Floating base delta [x, y, z, rz]
    [4:9]   Left arm (5D)
    [9:14]  Right arm (5D)
    [14]    Left gripper
    [15]    Right gripper

Proprioception Structure (16D for ACT):
    [0:4]   Floating base position [x, y, z, rz] - from proprioception_floating_base
    [4:9]   Left arm (qpos[1,2,3,4,13])
    [9:14]  Right arm (qpos[14,15,16,17,26])
    [14]    Left gripper - from proprioception_grippers[0]
    [15]    Right gripper - from proprioception_grippers[1]
"""

import h5py
import json
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF
from typing import Dict, List, Optional, Tuple
import logging

# Joint position indices from 60D proprioception (qpos only, indices 0-29)
# Note: New format has 60D (qpos 30 + qvel 30)
# Verified from robot._joints order:
# qpos[0-3]: left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow
# qpos[4-11]: left_gripper joints (8)
# qpos[12]: left_wrist
# qpos[13-16]: right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow
# qpos[17-24]: right_gripper joints (8)
# qpos[25]: right_wrist
# qpos[26-29]: pelvis_x, pelvis_y, pelvis_z, pelvis_rz
QPOS_LEFT_ARM = [0, 1, 2, 3, 12]       # 5D: shoulder_pitch, roll, yaw, elbow, wrist
QPOS_RIGHT_ARM = [13, 14, 15, 16, 25]  # 5D: shoulder_pitch, roll, yaw, elbow, wrist
QPOS_ARM_INDICES = QPOS_LEFT_ARM + QPOS_RIGHT_ARM  # 10D total

# Full ACT proprioception dimension: floating_base(4) + arms(10) + grippers(2) = 16
ACT_PROP_DIM = 16

# Action dimension (16D)
ACT_ACTION_DIM = 16


class ACTHdf5Dataset(Dataset):
    """
    PyTorch Dataset for ACT training with SaucepanToHob HDF5 data.
    
    Uses the new HDF5 format with:
    - proprioception_floating_base (direct, no integration needed)
    - proprioception_grippers (direct)
    - rgb_head (CHW format, 224x224)
    
    Args:
        hdf5_path: Path to demos.hdf5 file
        action_stats_path: Path to action_stats.json for normalization
        demo_ids: List of demo indices to use (None = all demos)
        image_size: Target image size (H, W)
        frame_stack: Number of frames to stack
        action_sequence: Number of future actions to predict (chunk size)
        normalize_actions: Whether to normalize actions to [-1, 1]
        min_max_margin: Margin for min-max normalization
        use_full_proprioception: If True, use full 60D. If False (default), use 16D action-relevant
        camera: Camera to use for RGB ('head', 'left_wrist', 'right_wrist')
        preload_images: If True, load all RGB images into memory (faster but uses more RAM)
        preload_all: If True, preload all data (images + proprioception + actions) into memory
    """
    
    def __init__(
        self,
        hdf5_path: str,
        action_stats_path: Optional[str] = None,
        demo_ids: Optional[List[int]] = None,
        image_size: Tuple[int, int] = (224, 224),
        frame_stack: int = 1,
        action_sequence: int = 16,
        normalize_actions: bool = True,
        min_max_margin: float = 0.0,
        use_full_proprioception: bool = False,
        camera: str = 'head',
        preload_images: bool = False,
        preload_all: bool = False,
    ):
        self.hdf5_path = Path(hdf5_path)
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.action_sequence = action_sequence
        self.normalize_actions = normalize_actions
        self.min_max_margin = min_max_margin
        self.use_full_proprioception = use_full_proprioception
        self.camera = camera
        self.rgb_key = f'rgb_{camera}'
        self.preload_images = preload_images
        self.preload_all = preload_all
        
        # Proprioception dimension
        if use_full_proprioception:
            self.prop_dim = 60  # New format: 60D (qpos 30 + qvel 30)
        else:
            self.prop_dim = ACT_PROP_DIM  # 16D
        
        # Load action stats for normalization
        self.action_stats = None
        if normalize_actions:
            self.action_stats = self._load_action_stats(action_stats_path)
        
        # Open HDF5 and build index
        self._load_data(demo_ids)
        
        # Preload data into memory if requested (reduces HDF5 I/O bottleneck)
        self._preloaded_rgb = None
        self._preloaded_prop = None
        self._preloaded_prop_fb = None
        self._preloaded_grippers = None
        self._preloaded_actions = None
        
        if preload_all or preload_images:
            self._preload_data(preload_all=preload_all)
        
        logging.info(f"ACTHdf5Dataset initialized:")
        logging.info(f"  - HDF5: {hdf5_path}")
        logging.info(f"  - Demos: {len(self.demo_keys)}")
        logging.info(f"  - Total samples: {len(self.sample_index)}")
        logging.info(f"  - Image size: {image_size}")
        logging.info(f"  - Camera: {camera}")
        logging.info(f"  - Action sequence: {action_sequence}")
        logging.info(f"  - Proprioception dim: {self.prop_dim} ({'full 60D' if use_full_proprioception else '16D action-relevant'})")
        logging.info(f"  - Preload images: {preload_images}")
        logging.info(f"  - Preload all: {preload_all}")
    
    def _preload_data(self, preload_all: bool = False):
        """Preload data into memory for faster training.
        
        This eliminates per-sample HDF5 I/O which is the main bottleneck.
        RGB images are the largest data and benefit most from preloading.
        
        Memory usage estimate (100 demos x 300 frames):
        - RGB: ~30K frames x 150KB = ~4.5GB
        - Proprioception: ~30K frames x 240B = ~7MB (negligible)
        - Actions: ~30K frames x 64B = ~2MB (negligible)
        """
        import time
        start_time = time.time()
        
        self._preloaded_rgb = {}
        total_frames = 0
        
        logging.info("Preloading data into memory...")
        
        with h5py.File(self.hdf5_path, 'r') as f:
            for demo_idx, demo_key in enumerate(self.demo_keys):
                demo = f[demo_key]
                
                # Always preload RGB (this is the main bottleneck)
                # Shape: (T+1, 3, 224, 224) uint8
                rgb_data = demo[self.rgb_key][:]  # Load all frames at once
                self._preloaded_rgb[demo_key] = rgb_data
                total_frames += rgb_data.shape[0]
                
                if preload_all:
                    # Also preload proprioception and actions
                    if self._preloaded_prop is None:
                        self._preloaded_prop = {}
                        self._preloaded_prop_fb = {}
                        self._preloaded_grippers = {}
                        self._preloaded_actions = {}
                    
                    self._preloaded_prop[demo_key] = demo['proprioception'][:]
                    self._preloaded_prop_fb[demo_key] = demo['proprioception_floating_base'][:]
                    self._preloaded_grippers[demo_key] = demo['proprioception_grippers'][:]
                    self._preloaded_actions[demo_key] = demo['actions'][:]
                
                if (demo_idx + 1) % 10 == 0:
                    logging.info(f"  Preloaded {demo_idx + 1}/{len(self.demo_keys)} demos")
        
        elapsed = time.time() - start_time
        memory_gb = total_frames * 3 * 224 * 224 / (1024**3)  # RGB memory estimate
        logging.info(f"Preload complete: {total_frames} frames, ~{memory_gb:.2f}GB RGB, {elapsed:.1f}s")

    def _load_action_stats(self, stats_path: Optional[str]) -> Dict:
        """Load action statistics for normalization.
        
        Supports multiple formats:
        1. Flat format: {min: [16], max: [16], ...}
        2. 'full' key format: {full: {min: [16], max: [16], ...}}
        3. BRS format: {mobile_base: {...}, torso: {...}, arms: {...}}
        4. Legacy ACT format: {floating_base: {...}, arms: {...}}
        """
        if stats_path is None:
            stats_path = self.hdf5_path.parent / 'action_stats.json'
        else:
            stats_path = Path(stats_path)
        
        if not stats_path.exists():
            logging.warning(f"Action stats not found: {stats_path}, normalization disabled")
            return None
        
        with open(stats_path, 'r') as f:
            raw_stats = json.load(f)
        
        # Format 1: Flat format with direct arrays
        if 'min' in raw_stats and isinstance(raw_stats['min'], list) and len(raw_stats['min']) == 16:
            return {
                'min': np.array(raw_stats['min'], dtype=np.float32),
                'max': np.array(raw_stats['max'], dtype=np.float32),
                'mean': np.array(raw_stats['mean'], dtype=np.float32),
                'std': np.array(raw_stats['std'], dtype=np.float32),
            }
        
        # Format 2: 'full' key format (BRS style)
        if 'full' in raw_stats:
            return {
                'min': np.array(raw_stats['full']['min'], dtype=np.float32),
                'max': np.array(raw_stats['full']['max'], dtype=np.float32),
                'mean': np.array(raw_stats['full']['mean'], dtype=np.float32),
                'std': np.array(raw_stats['full']['std'], dtype=np.float32),
            }
        
        # Helper function
        def extend_or_append(target, value):
            if isinstance(value, list):
                target.extend(value)
            else:
                target.append(value)
        
        min_list, max_list, mean_list, std_list = [], [], [], []
        
        # Format 3: BRS format (mobile_base + torso + arms)
        # BRS 16D: [mobile_base(3), torso(1), arms(12)]
        # ACT needs BigYM 16D: [floating_base(4), left_arm(5), right_arm(5), grippers(2)]
        if 'mobile_base' in raw_stats and 'torso' in raw_stats:
            # Convert BRS format to BigYM format
            # BRS: [mobile_base(3): dx,dy,drz] [torso(1): dz] [arms(12): left_arm(5)+left_grip(1)+right_arm(5)+right_grip(1)]
            # BigYM: [fb(4): dx,dy,dz,drz] [left_arm(5)] [right_arm(5)] [grippers(2)]
            
            mb = raw_stats['mobile_base']
            torso = raw_stats['torso']
            arms = raw_stats['arms']
            
            # floating_base: [dx, dy, dz, drz]
            # From BRS: mobile_base[0,1], torso, mobile_base[2]
            min_list.extend([mb['min'][0], mb['min'][1]])
            min_list.append(torso['min'] if not isinstance(torso['min'], list) else torso['min'][0])
            min_list.append(mb['min'][2])
            
            max_list.extend([mb['max'][0], mb['max'][1]])
            max_list.append(torso['max'] if not isinstance(torso['max'], list) else torso['max'][0])
            max_list.append(mb['max'][2])
            
            mean_list.extend([mb['mean'][0], mb['mean'][1]])
            mean_list.append(torso['mean'] if not isinstance(torso['mean'], list) else torso['mean'][0])
            mean_list.append(mb['mean'][2])
            
            std_list.extend([mb['std'][0], mb['std'][1]])
            std_list.append(torso['std'] if not isinstance(torso['std'], list) else torso['std'][0])
            std_list.append(mb['std'][2])
            
            # arms: BRS has [left_arm(5), left_grip(1), right_arm(5), right_grip(1)]
            # BigYM wants [left_arm(5), right_arm(5), grippers(2)]
            # BRS indices: 0-4=left_arm, 5=left_grip, 6-10=right_arm, 11=right_grip
            min_list.extend(arms['min'][0:5])   # left_arm
            min_list.extend(arms['min'][6:11])  # right_arm
            min_list.append(arms['min'][5])     # left_gripper
            min_list.append(arms['min'][11])    # right_gripper
            
            max_list.extend(arms['max'][0:5])
            max_list.extend(arms['max'][6:11])
            max_list.append(arms['max'][5])
            max_list.append(arms['max'][11])
            
            mean_list.extend(arms['mean'][0:5])
            mean_list.extend(arms['mean'][6:11])
            mean_list.append(arms['mean'][5])
            mean_list.append(arms['mean'][11])
            
            std_list.extend(arms['std'][0:5])
            std_list.extend(arms['std'][6:11])
            std_list.append(arms['std'][5])
            std_list.append(arms['std'][11])
            
            logging.info("  Loaded action stats (BRS format -> BigYM 16D)")
        
        # Format 4: Legacy ACT format (floating_base + arms)
        elif 'floating_base' in raw_stats:
            extend_or_append(min_list, raw_stats['floating_base']['min'])
            extend_or_append(max_list, raw_stats['floating_base']['max'])
            extend_or_append(mean_list, raw_stats['floating_base']['mean'])
            extend_or_append(std_list, raw_stats['floating_base']['std'])
            
            if 'arms' in raw_stats:
                extend_or_append(min_list, raw_stats['arms']['min'])
                extend_or_append(max_list, raw_stats['arms']['max'])
                extend_or_append(mean_list, raw_stats['arms']['mean'])
                extend_or_append(std_list, raw_stats['arms']['std'])
            
            logging.info("  Loaded action stats (Legacy ACT format)")
        
        else:
            logging.warning(f"Unknown action stats format, keys: {list(raw_stats.keys())}")
            return None
        
        return {
            'min': np.array(min_list, dtype=np.float32),
            'max': np.array(max_list, dtype=np.float32),
            'mean': np.array(mean_list, dtype=np.float32),
            'std': np.array(std_list, dtype=np.float32),
        }
    
    def _load_data(self, demo_ids: Optional[List[int]]):
        """Load HDF5 file and build sample index."""
        with h5py.File(self.hdf5_path, 'r') as f:
            # Get all demo keys
            all_keys = sorted([k for k in f.keys() if k.startswith('demo_')],
                             key=lambda x: int(x.split('_')[1]))
            
            if demo_ids is not None:
                self.demo_keys = [f'demo_{i}' for i in demo_ids if f'demo_{i}' in f]
            else:
                self.demo_keys = all_keys
            
            # Build sample index: (demo_key, timestep)
            self.sample_index = []
            self.demo_lengths = {}
            
            for demo_key in self.demo_keys:
                # actions shape: (T, 16)
                # observations shape: (T+1, ...)
                n_actions = f[demo_key]['actions'].shape[0]
                self.demo_lengths[demo_key] = n_actions
                
                # Valid start indices:
                # - Need frame_stack-1 frames before current (for frame stacking)
                # - Need action_sequence actions after current (for action prediction)
                # - Observation at timestep t corresponds to state before action[t]
                start_idx = self.frame_stack - 1
                end_idx = n_actions - self.action_sequence + 1
                
                for t in range(start_idx, end_idx):
                    self.sample_index.append((demo_key, t))
        
        # Per-worker HDF5 file handles (for multi-worker DataLoader)
        self._hdf5_file = None
        self._worker_id = None
    
    def _get_hdf5(self):
        """Get HDF5 file handle (open if needed, per-worker)."""
        # Check if we need to open a new handle (different worker)
        worker_info = torch.utils.data.get_worker_info()
        current_worker = worker_info.id if worker_info else -1
        
        if self._hdf5_file is None or self._worker_id != current_worker:
            if self._hdf5_file is not None:
                self._hdf5_file.close()
            # Open with rdcc (chunk cache) settings for better performance
            self._hdf5_file = h5py.File(
                self.hdf5_path, 'r',
                rdcc_nbytes=256*1024*1024,  # 256MB chunk cache
                rdcc_nslots=10007,  # Prime number for hash table
            )
            self._worker_id = current_worker
        return self._hdf5_file
    
    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        """Normalize action to [-1, 1] using min-max normalization."""
        if self.action_stats is None:
            return action
        
        min_val = self.action_stats['min']
        max_val = self.action_stats['max']
        
        # Add margin
        range_val = max_val - min_val
        min_val = min_val - self.min_max_margin * range_val
        max_val = max_val + self.min_max_margin * range_val
        
        # Normalize to [-1, 1]
        normalized = 2.0 * (action - min_val) / (max_val - min_val + 1e-8) - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
    
    def __len__(self):
        return len(self.sample_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        demo_key, t = self.sample_index[idx]
        
        # Use preloaded data if available, otherwise read from HDF5
        use_preloaded_rgb = self._preloaded_rgb is not None
        use_preloaded_all = self._preloaded_actions is not None
        
        # Only open HDF5 if we need to read from it
        demo = None
        if not use_preloaded_rgb or not use_preloaded_all:
            f = self._get_hdf5()
            demo = f[demo_key]
        
        # Load RGB images with frame stacking
        # New format: rgb_head shape is (T+1, 3, 224, 224) - already CHW!
        rgb_frames = []
        for i in range(self.frame_stack):
            frame_idx = t - (self.frame_stack - 1) + i
            
            if use_preloaded_rgb:
                rgb = self._preloaded_rgb[demo_key][frame_idx]  # (3, 224, 224) uint8
            else:
                rgb = demo[self.rgb_key][frame_idx]  # (3, 224, 224) uint8
            
            # Convert to float tensor
            rgb_tensor = torch.from_numpy(rgb.astype(np.float32))  # (3, 224, 224)
            
            # Resize if needed
            if rgb_tensor.shape[1:] != self.image_size:
                rgb_tensor = TF.resize(rgb_tensor, self.image_size, antialias=True)
            
            rgb_frames.append(rgb_tensor)
        
        # Stack frames: (frame_stack, 3, H, W)
        rgb_stacked = torch.stack(rgb_frames, dim=0)
        
        # Load proprioception
        prop_frames = []
        for i in range(self.frame_stack):
            frame_idx = t - (self.frame_stack - 1) + i
            
            if self.use_full_proprioception:
                # Use full 60D proprioception
                if use_preloaded_all:
                    prop = self._preloaded_prop[demo_key][frame_idx]  # (60,)
                else:
                    prop = demo['proprioception'][frame_idx]  # (60,)
            else:
                # Construct 16D action-relevant proprioception:
                # [0:4] = floating base position (from proprioception_floating_base)
                # [4:14] = arm joint positions (from proprioception qpos)
                # [14:16] = grippers (from proprioception_grippers)
                
                if use_preloaded_all:
                    fb_pos = self._preloaded_prop_fb[demo_key][frame_idx]  # (4,)
                    prop_full = self._preloaded_prop[demo_key][frame_idx]  # (60,)
                    grippers = self._preloaded_grippers[demo_key][frame_idx]  # (2,)
                else:
                    fb_pos = demo['proprioception_floating_base'][frame_idx]  # (4,)
                    prop_full = demo['proprioception'][frame_idx]  # (60,)
                    grippers = demo['proprioception_grippers'][frame_idx]  # (2,)
                
                arm_qpos = prop_full[QPOS_ARM_INDICES]  # (10,) - arm joints only
                prop = np.concatenate([fb_pos, arm_qpos, grippers])  # (16,)
            
            prop_frames.append(torch.from_numpy(prop.astype(np.float32)))
        
        # Stack proprioception: (frame_stack, prop_dim) -> flatten
        prop_stacked = torch.stack(prop_frames, dim=0)
        low_dim_state = prop_stacked.flatten()
        
        # Load action sequence
        actions = []
        is_pad = []
        n_actions = self.demo_lengths[demo_key]
        
        for i in range(self.action_sequence):
            action_idx = t + i
            if action_idx < n_actions:
                if use_preloaded_all:
                    action = self._preloaded_actions[demo_key][action_idx]  # (16,)
                else:
                    action = demo['actions'][action_idx]  # (16,)
                if self.normalize_actions:
                    action = self._normalize_action(action)
                actions.append(torch.from_numpy(action.astype(np.float32)))
                is_pad.append(False)
            else:
                # Padding with zeros
                actions.append(torch.zeros(ACT_ACTION_DIM, dtype=torch.float32))
                is_pad.append(True)
        
        # Stack actions: (action_sequence, 16)
        action_tensor = torch.stack(actions, dim=0)
        is_pad_tensor = torch.tensor(is_pad, dtype=torch.bool)
        
        return {
            'rgb_head': rgb_stacked,  # (frame_stack, 3, H, W)
            'low_dim_state': low_dim_state,  # (frame_stack * prop_dim,)
            'action': action_tensor,  # (action_sequence, 16)
            'is_pad': is_pad_tensor,  # (action_sequence,)
        }
    
    def close(self):
        """Close HDF5 file."""
        if self._hdf5_file is not None:
            self._hdf5_file.close()
            self._hdf5_file = None
    
    def __del__(self):
        self.close()


class ACTHdf5DataModule:
    """
    Data module for ACT training with HDF5 dataset.
    
    Handles train/val split and DataLoader creation.
    """
    
    def __init__(
        self,
        hdf5_path: str,
        action_stats_path: Optional[str] = None,
        val_ratio: float = 0.1,
        batch_size: int = 64,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (224, 224),
        frame_stack: int = 1,
        action_sequence: int = 16,
        normalize_actions: bool = True,
        min_max_margin: float = 0.0,
        use_full_proprioception: bool = False,
        camera: str = 'head',
        preload_images: bool = False,
        preload_all: bool = False,
    ):
        self.hdf5_path = hdf5_path
        self.action_stats_path = action_stats_path
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.action_sequence = action_sequence
        self.normalize_actions = normalize_actions
        self.min_max_margin = min_max_margin
        self.use_full_proprioception = use_full_proprioception
        self.camera = camera
        self.preload_images = preload_images
        self.preload_all = preload_all
        
        self.train_dataset = None
        self.val_dataset = None
    
    def setup(self):
        """Create train/val datasets."""
        # Get total number of demos
        with h5py.File(self.hdf5_path, 'r') as f:
            demo_keys = sorted([k for k in f.keys() if k.startswith('demo_')],
                              key=lambda x: int(x.split('_')[1]))
            n_demos = len(demo_keys)
        
        # Split demos
        n_val = max(1, int(n_demos * self.val_ratio))
        n_train = n_demos - n_val
        
        train_ids = list(range(n_train))
        val_ids = list(range(n_train, n_demos))
        
        logging.info(f"Dataset split: {n_train} train, {n_val} val demos")
        
        # Create datasets
        # Note: When preloading, num_workers should be 0 as data is already in memory
        # and workers would duplicate the preloaded data
        self.train_dataset = ACTHdf5Dataset(
            hdf5_path=self.hdf5_path,
            action_stats_path=self.action_stats_path,
            demo_ids=train_ids,
            image_size=self.image_size,
            frame_stack=self.frame_stack,
            action_sequence=self.action_sequence,
            normalize_actions=self.normalize_actions,
            min_max_margin=self.min_max_margin,
            use_full_proprioception=self.use_full_proprioception,
            camera=self.camera,
            preload_images=self.preload_images,
            preload_all=self.preload_all,
        )
        
        self.val_dataset = ACTHdf5Dataset(
            hdf5_path=self.hdf5_path,
            action_stats_path=self.action_stats_path,
            demo_ids=val_ids,
            image_size=self.image_size,
            frame_stack=self.frame_stack,
            action_sequence=self.action_sequence,
            normalize_actions=self.normalize_actions,
            min_max_margin=self.min_max_margin,
            use_full_proprioception=self.use_full_proprioception,
            camera=self.camera,
            preload_images=self.preload_images,
            preload_all=self.preload_all,
        )
    
    def train_dataloader(self) -> DataLoader:
        # When preloading, reduce workers to 0 since data is in memory
        # Multi-worker would duplicate preloaded data in memory
        num_workers = 0 if (self.preload_images or self.preload_all) else self.num_workers
        
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        num_workers = 0 if (self.preload_images or self.preload_all) else self.num_workers
        
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=4 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )


if __name__ == '__main__':
    # Test the dataset with new HDF5 format
    logging.basicConfig(level=logging.INFO)
    
    hdf5_path = '/home/hyunjin/bigym_ws/data/demonstrations/saucepan_to_hob/demos.hdf5'
    
    print("Testing ACTHdf5Dataset with new HDF5 format...")
    
    dataset = ACTHdf5Dataset(
        hdf5_path=hdf5_path,
        image_size=(224, 224),
        frame_stack=1,
        action_sequence=16,
        normalize_actions=False,  # Skip normalization for testing
    )
    
    print(f"\nTotal samples: {len(dataset)}")
    
    # Test loading a sample
    sample = dataset[0]
    print(f"\nSample shapes:")
    for k, v in sample.items():
        print(f"  {k}: {v.shape} {v.dtype}")
    
    # Print proprioception details
    print(f"\nProprioception (16D):")
    prop = sample['low_dim_state']
    print(f"  [0:4] floating_base: {prop[0:4].numpy()}")
    print(f"  [4:9] left_arm: {prop[4:9].numpy()}")
    print(f"  [9:14] right_arm: {prop[9:14].numpy()}")
    print(f"  [14:16] grippers: {prop[14:16].numpy()}")
    
    # Test DataLoader
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape} {v.dtype}")
    
    dataset.close()
    print("\nTest passed!")
