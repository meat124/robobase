"""
BRS Policy Training with PyTorch Lightning
Using brs-algo modules directly
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# Import modules from brs-algo
# ============================================================================
# Add brs-algo to path
brs_algo_path = Path(__file__).parent.parent.parent.parent / "brs-algo"
sys.path.insert(0, str(brs_algo_path))

# Import from brs-algo
from brs_algo.learning.policy import WBVIMAPolicy, WBVIMAPolicyRGB
from brs_algo.learning.module import DiffusionModule
from brs_algo.learning.data import ActionSeqChunkDataModule
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# Import PCD dataset classes
from robobase.method.pcd_dataset import PCDBRSDataset, PCDDataModule, pcd_brs_collate_fn
from robobase.method.rgb_dataset import RGBBRSDataset, RGBDataModule, rgb_brs_collate_fn
from robobase.method.rollout_callback import RolloutEvaluationCallback


class SafeDiffusionModule(DiffusionModule):
    """
    Wrapper around DiffusionModule that adds NaN checking to prevent silent failures.
    When NaN is detected in loss, logs a warning and returns a safe value instead of 
    propagating NaN through the gradient computation.
    """
    
    def training_step(self, batch, batch_idx):
        """Override training_step to add NaN detection"""
        loss, log_dict, batch_size = self.imitation_training_step(batch, batch_idx)
        
        # Check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️  WARNING: NaN/Inf detected in training loss at batch {batch_idx}")
            print(f"    Log dict: {log_dict}")
            # Return a safe loss value to prevent gradient explosion
            # This is a stopgap - the model likely needs to be retrained with different settings
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
            log_dict = {k: torch.tensor(0.0) if torch.isnan(v) or torch.isinf(v) else v 
                       for k, v in log_dict.items()}
        
        log_dict = {f"train/{k}": v for k, v in log_dict.items()}
        log_dict["train/loss"] = loss
        self.log_dict(
            log_dict,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        return loss


# base 3 torso 1 left arm 5 left gripper 1 right arm 5 right gripper 1
class DummyBRSDataset(Dataset):
    """
    Dummy dataset for testing BRS training
    Generates random observations and actions
    """
    def __init__(
        self,
        num_samples: int,
        num_latest_obs: int,
        action_prediction_horizon: int,
        prop_dim: int,
        n_points: int,
    ):
        self.num_samples = num_samples
        self.num_latest_obs = num_latest_obs
        self.action_prediction_horizon = action_prediction_horizon
        self.prop_dim = prop_dim
        self.n_points = n_points
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Generate dummy observation following brs-algo structure
        # Observations have shape: (num_latest_obs, ...)
        
        # Generate random pointcloud data
        pcd_data = np.random.randn(self.num_latest_obs, self.n_points, 6).astype(np.float32)
        
        obs = {
            "odom": {
                "base_velocity": np.random.randn(self.num_latest_obs, 3).astype(np.float32),
            },
            "qpos": {
                "torso": np.random.randn(self.num_latest_obs, 1).astype(np.float32),
                "left_arm": np.random.randn(self.num_latest_obs, 5).astype(np.float32),
                "left_gripper": np.random.randn(self.num_latest_obs, 1).astype(np.float32),
                "right_arm": np.random.randn(self.num_latest_obs, 5).astype(np.float32),
                "right_gripper": np.random.randn(self.num_latest_obs, 1).astype(np.float32),
            },
            "pointcloud": {
                "xyz": pcd_data[..., :3],  # (num_latest_obs, n_points, 3)
                "rgb": pcd_data[..., 3:],  # (num_latest_obs, n_points, 3)
            }
        }
        
        # Generate dummy actions matching action_keys (16 dims total)
        # Actions have shape: (num_latest_obs, action_prediction_horizon, dim)
        action_chunks = {
            "mobile_base": np.random.randn(self.num_latest_obs, self.action_prediction_horizon, 3).astype(np.float32),
            "torso": np.random.randn(self.num_latest_obs, self.action_prediction_horizon, 1).astype(np.float32),
            "arms": np.random.randn(self.num_latest_obs, self.action_prediction_horizon, 12).astype(np.float32),
        }
        
        # Padding mask: all ones (no padding)
        pad_mask = np.ones((self.num_latest_obs, self.action_prediction_horizon), dtype=np.float32)
        
        return {
            **obs,  # Flatten obs dict into main dict
            "action_chunks": action_chunks,
            "pad_mask": pad_mask,
        }


def brs_collate_fn(batch):
    """
    Collate function for BRS dataset
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
            "torso": qpos_torso.unsqueeze(0),  # (1, B, window_size, 4)
            "left_arm": qpos_left_arm.unsqueeze(0),
            "left_gripper": qpos_left_gripper.unsqueeze(0),
            "right_arm": qpos_right_arm.unsqueeze(0),
            "right_gripper": qpos_right_gripper.unsqueeze(0),
        },
        "pointcloud": {
            "xyz": pcd_xyz.unsqueeze(0),  # (1, B, window_size, n_points, 3)
            "rgb": pcd_rgb.unsqueeze(0),
        },
        "action_chunks": {k: v.unsqueeze(0) for k, v in action_chunks.items()},  # (1, B, window_size, action_horizon, dim)
        "pad_mask": pad_mask.unsqueeze(0),  # (1, B, window_size, action_horizon)
    }


# ============================================================================
# PyTorch Lightning DataModule
# ============================================================================

class DummyDataModule(pl.LightningDataModule):
    """
    DataModule for dummy data (for testing)
    In practice, replace with ActionSeqChunkDataModule
    """
    def __init__(
        self,
        num_latest_obs: int,
        action_prediction_horizon: int,
        prop_dim: int,
        pcd_downsample_points: int,
        batch_size: int,
        val_batch_size: int,
        val_split_ratio: float,
        dataloader_num_workers: int,
        prefetch_factor: int = 4,
        persistent_workers: bool = True,
        seed: Optional[int] = None,
    ):
        super().__init__()
        self.num_latest_obs = num_latest_obs
        self.action_prediction_horizon = action_prediction_horizon
        self.prop_dim = prop_dim
        self.pcd_downsample_points = pcd_downsample_points
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.val_split_ratio = val_split_ratio
        self.dataloader_num_workers = dataloader_num_workers
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers
        self.seed = seed
    
    def setup(self, stage: Optional[str] = None):
        # Create dummy datasets
        num_total = 1200
        num_val = int(num_total * self.val_split_ratio)
        num_train = num_total - num_val
        
        self.train_dataset = DummyBRSDataset(
            num_samples=num_train,
            num_latest_obs=self.num_latest_obs,
            action_prediction_horizon=self.action_prediction_horizon,
            prop_dim=self.prop_dim,
            n_points=self.pcd_downsample_points
        )
        
        self.val_dataset = DummyBRSDataset(
            num_samples=num_val,
            num_latest_obs=self.num_latest_obs,
            action_prediction_horizon=self.action_prediction_horizon,
            prop_dim=self.prop_dim,
            n_points=self.pcd_downsample_points
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.dataloader_num_workers,
            collate_fn=brs_collate_fn,
            prefetch_factor=self.prefetch_factor if self.dataloader_num_workers > 0 else None,
            persistent_workers=self.persistent_workers and self.dataloader_num_workers > 0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
            collate_fn=brs_collate_fn,
            prefetch_factor=self.prefetch_factor if self.dataloader_num_workers > 0 else None,
            persistent_workers=self.persistent_workers and self.dataloader_num_workers > 0
        )


# ============================================================================
# Config Loading
# ============================================================================


def expand_path(path_str: str) -> str:
    """
    Expand path string to absolute path
    - Expands ~ to home directory
    - Resolves relative paths to absolute paths
    """
    if path_str is None:
        return None
    
    # Expand ~ to home directory
    path = Path(path_str).expanduser()
    
    # If relative path, resolve to absolute
    if not path.is_absolute():
        # Resolve relative to current working directory
        path = Path.cwd() / path
    
    return str(path.resolve())


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file and expand paths"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Expand paths in config
    if 'hdf5_path' in config and config['hdf5_path']:
        config['hdf5_path'] = expand_path(config['hdf5_path'])
    if 'pcd_root' in config and config['pcd_root']:
        config['pcd_root'] = expand_path(config['pcd_root'])
    
    return config


def create_policy_from_config(config: Dict[str, Any]) -> WBVIMAPolicy:
    """Create WBVIMAPolicy from config"""
    
    # Create noise scheduler
    noise_scheduler_config = config['noise_scheduler']
    noise_scheduler = DDIMScheduler(**noise_scheduler_config)
    
    policy = WBVIMAPolicy(
        prop_dim=config['prop_dim'],
        prop_keys=config['prop_keys'],
        prop_mlp_hidden_depth=config['prop_mlp_hidden_depth'],
        prop_mlp_hidden_dim=config['prop_mlp_hidden_dim'],
        pointnet_n_coordinates=config['pointnet_n_coordinates'],
        pointnet_n_color=config['pointnet_n_color'],
        pointnet_hidden_depth=config['pointnet_hidden_depth'],
        pointnet_hidden_dim=config['pointnet_hidden_dim'],
        num_latest_obs=config['num_latest_obs'],
        use_modality_type_tokens=config['use_modality_type_tokens'],
        xf_n_embd=config['xf_n_embd'],
        xf_n_layer=config['xf_n_layer'],
        xf_n_head=config['xf_n_head'],
        xf_dropout_rate=config['xf_dropout_rate'],
        xf_use_geglu=config['xf_use_geglu'],
        learnable_action_readout_token=config['learnable_action_readout_token'],
        action_dim=config['action_dim'],
        action_prediction_horizon=config['action_prediction_horizon'],
        diffusion_step_embed_dim=config['diffusion_step_embed_dim'],
        unet_down_dims=config['unet_down_dims'],
        unet_kernel_size=config['unet_kernel_size'],
        unet_n_groups=config['unet_n_groups'],
        unet_cond_predict_scale=config['unet_cond_predict_scale'],
        action_keys=config['action_keys'],
        action_key_dims=config['action_key_dims'],
        noise_scheduler=noise_scheduler,
        noise_scheduler_step_kwargs=config['noise_scheduler_step_kwargs'],
        num_denoise_steps_per_inference=config['num_denoise_steps_per_inference'],
    )
    
    return policy


def create_rgb_policy_from_config(config: Dict[str, Any]) -> WBVIMAPolicyRGB:
    """Create WBVIMAPolicyRGB from config (RGB version using ResNet18)"""
    
    # Create noise scheduler
    noise_scheduler_config = config['noise_scheduler']
    noise_scheduler = DDIMScheduler(**noise_scheduler_config)
    
    policy = WBVIMAPolicyRGB(
        prop_dim=config['prop_dim'],
        prop_keys=config['prop_keys'],
        prop_mlp_hidden_depth=config['prop_mlp_hidden_depth'],
        prop_mlp_hidden_dim=config['prop_mlp_hidden_dim'],
        resnet_pretrained=config.get('resnet_pretrained', True),
        resnet_freeze_backbone=config.get('resnet_freeze_backbone', False),
        num_camera_views=config.get('num_camera_views', 1),
        num_latest_obs=config['num_latest_obs'],
        use_modality_type_tokens=config['use_modality_type_tokens'],
        xf_n_embd=config['xf_n_embd'],
        xf_n_layer=config['xf_n_layer'],
        xf_n_head=config['xf_n_head'],
        xf_dropout_rate=config['xf_dropout_rate'],
        xf_use_geglu=config['xf_use_geglu'],
        learnable_action_readout_token=config['learnable_action_readout_token'],
        action_dim=config['action_dim'],
        action_prediction_horizon=config['action_prediction_horizon'],
        diffusion_step_embed_dim=config['diffusion_step_embed_dim'],
        unet_down_dims=config['unet_down_dims'],
        unet_kernel_size=config['unet_kernel_size'],
        unet_n_groups=config['unet_n_groups'],
        unet_cond_predict_scale=config['unet_cond_predict_scale'],
        action_keys=config['action_keys'],
        action_key_dims=config['action_key_dims'],
        noise_scheduler=noise_scheduler,
        noise_scheduler_step_kwargs=config['noise_scheduler_step_kwargs'],
        num_denoise_steps_per_inference=config['num_denoise_steps_per_inference'],
    )
    
    return policy


def create_module_from_config(config: Dict[str, Any], use_rgb: bool = False) -> SafeDiffusionModule:
    """Create SafeDiffusionModule from config (with NaN detection)
    
    Args:
        config: Configuration dictionary
        use_rgb: If True, create RGB policy (ResNet18), else PCD policy (PointNet)
    """
    if use_rgb:
        policy = create_rgb_policy_from_config(config)
    else:
        policy = create_policy_from_config(config)
    
    module = SafeDiffusionModule(
        policy=policy,
        action_prediction_horizon=config['action_prediction_horizon'],
        lr=config['lr'],
        use_cosine_lr=config['use_cosine_lr'],
        lr_warmup_steps=config['lr_warmup_steps'],
        lr_cosine_steps=config['lr_cosine_steps'],
        lr_cosine_min=config['lr_cosine_min'],
        lr_layer_decay=config['lr_layer_decay'],
        weight_decay=config['wd'],
        action_keys=config['action_keys'],
        loss_on_latest_obs_only=config['loss_on_latest_obs_only'],
    )
    
    return module



def create_data_module_from_config(config: Dict[str, Any], use_dummy: bool = True, use_pcd: bool = False, use_rgb: bool = False):
    """
    Create DataModule from config
    
    Args:
        config: Config dict
        use_dummy: If True, use DummyDataModule. If False, use real data
        use_pcd: If True, use PCDDataModule with PCD files
        use_rgb: If True, use RGBDataModule with RGB images
    """
    if use_dummy:
        data_module = DummyDataModule(
            num_latest_obs=config['num_latest_obs'],
            action_prediction_horizon=config['action_prediction_horizon'],
            batch_size=config['bs'],
            prop_dim=config['prop_dim'],
            pcd_downsample_points=config.get('pcd_downsample_points', 4096),
            val_batch_size=config['vbs'],
            val_split_ratio=config['val_split_ratio'],
            dataloader_num_workers=config['dataloader_num_workers'],
            prefetch_factor=config.get('prefetch_factor', 4),
            persistent_workers=config.get('persistent_workers', True),
            seed=config['seed'] if config['seed'] > 0 else None,
        )
    elif use_rgb:
        # Use RGB data module
        data_module = RGBDataModule(
            hdf5_path=config['hdf5_path'],
            cameras=config.get('cameras', ['head']),
            num_latest_obs=config['num_latest_obs'],
            action_prediction_horizon=config['action_prediction_horizon'],
            image_size=tuple(config.get('image_size', [224, 224])),
            action_stats_path=config.get('action_stats_path', None),
            prop_stats_path=config.get('prop_stats_path', None),
            normalize=config.get('normalize', True),
            batch_size=config['bs'],
            val_batch_size=config['vbs'],
            num_workers=config['dataloader_num_workers'],
            val_split_ratio=config['val_split_ratio'],
            preload_data=config.get('preload_data', False),
        )
    elif use_pcd:
        # Use PCD data module
        data_module = PCDDataModule(
            hdf5_path=config['hdf5_path'],
            pcd_root=config['pcd_root'],
            demo_ids=config.get('demo_ids', None),  # Auto-discover if not provided
            cameras=config.get('cameras', ['head']),
            num_latest_obs=config['num_latest_obs'],
            action_prediction_horizon=config['action_prediction_horizon'],
            max_points_per_camera=config.get('max_points_per_camera', 4096),
            batch_size=config['bs'],
            val_batch_size=config['vbs'],
            val_split_ratio=config['val_split_ratio'],
            dataloader_num_workers=config['dataloader_num_workers'],
            prefetch_factor=config.get('prefetch_factor', 4),
            persistent_workers=config.get('persistent_workers', True),
            seed=config['seed'] if config['seed'] > 0 else None,
            normalize=config.get('normalize', True),
            action_stats_path=config.get('action_stats_path', None),
            prop_stats_path=config.get('prop_stats_path', None),
            pcd_stats_path=config.get('pcd_stats_path', None),
            normalize_pcd=config.get('normalize_pcd', True),  # Separate flag for PCD normalization
            subsample_points=config.get('subsample_points', True),  # Control runtime downsampling
            preload_hdf5=config.get('preload_hdf5', False),
            preload_pcd=config.get('preload_pcd', False),
        )
    else:
        # Use real data module from brs-algo
        # This requires data_path to be set in config
        data_module = ActionSeqChunkDataModule(
            data_path=config['data_path'],
            pcd_downsample_points=config['pcd_downsample_points'],
            pcd_x_range=config.get('pcd_x_range', (-2.0, 2.0)),
            pcd_y_range=config.get('pcd_y_range', (-2.0, 2.0)),
            pcd_z_range=config.get('pcd_z_range', (0.0, 2.0)),
            mobile_base_vel_action_min=config.get('mobile_base_vel_action_min', (-1.0, -1.0, -1.0)),
            mobile_base_vel_action_max=config.get('mobile_base_vel_action_max', (1.0, 1.0, 1.0)),
            load_visual_obs_in_memory=config.get('load_visual_obs_in_memory', True),
            multi_view_cameras=config.get('multi_view_cameras', None),
            load_multi_view_camera_rgb=config.get('load_multi_view_camera_rgb', False),
            load_multi_view_camera_depth=config.get('load_multi_view_camera_depth', False),
            obs_window_size=config['num_latest_obs'],
            action_prediction_horizon=config['action_prediction_horizon'],
            batch_size=config['bs'],
            val_batch_size=config['vbs'],
            val_split_ratio=config['val_split_ratio'],
            dataloader_num_workers=config['dataloader_num_workers'],
            seed=config['seed'] if config['seed'] > 0 else None,
        )
    
    return data_module



# ============================================================================
# Main Training Function
# ============================================================================

def train(config_path: str, use_pcd: bool = False, use_rgb: bool = False, **overrides):
    """
    Main training function following brs-algo structure
    
    Args:
        config_path: Path to config YAML file
        use_pcd: If True, use PCDDataModule with PointNet encoder
        use_rgb: If True, use RGBDataModule with ResNet18 encoder
        **overrides: Override config values
    """
    # Validation: can't use both PCD and RGB
    if use_pcd and use_rgb:
        raise ValueError("Cannot use both --use-pcd and --use-rgb. Choose one.")
    
    # Load config
    config = load_config(config_path)
    
    # Apply overrides
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    
    # Set seed
    if config['seed'] > 0:
        pl.seed_everything(config['seed'], workers=True)
    
    # Determine data mode string
    if use_rgb:
        data_mode = "RGB Dataset (ResNet18)"
    elif use_pcd:
        data_mode = "PCD Dataset (PointNet)"
    else:
        data_mode = "Dummy Dataset"
    
    # Print config
    print("="*80)
    print("BRS Policy Training (PyTorch Lightning)")
    print("="*80)
    print(f"Run name: {config['wandb_name']}")
    print(f"Config: {config_path}")
    print(f"Data mode: {data_mode}")
    if use_pcd:
        print(f"HDF5 path: {config.get('hdf5_path', 'NOT SET')}")
        print(f"PCD root: {config.get('pcd_root', 'NOT SET')}")
    elif use_rgb:
        print(f"HDF5 path: {config.get('hdf5_path', 'NOT SET')}")
        print(f"Cameras: {config.get('cameras', ['head'])}")
        print(f"Image size: {config.get('image_size', [224, 224])}")
    print(f"Batch size: {config['bs']} (val: {config['vbs']})")
    print(f"Learning rate: {config['lr']}")
    print(f"Weight decay: {config['wd']}")
    print(f"Num latest obs: {config['num_latest_obs']}")
    print(f"Action prediction horizon: {config['action_prediction_horizon']}")
    print()
    
    # Create run directory - use wandb_name for consistency with wandb logging
    wandb_name = config.get('wandb_name', config.get('wandb_run_name', config['wandb_name']))
    run_dir = Path("runs") / wandb_name
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run directory: {run_dir}")
    
    # Save config
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Store config for callbacks
    trainer_config = config.copy()
    
    # Copy stats files to run directory if using PCD dataset
    if use_pcd:
        import shutil
        
        # Get stats paths from config or use defaults
        hdf5_path = Path(config.get('hdf5_path', ''))
        action_stats_path = config.get('action_stats_path', None)
        prop_stats_path = config.get('prop_stats_path', None)
        pcd_stats_path = config.get('pcd_stats_path', None)
        
        # Use default paths if not specified
        if action_stats_path is None and hdf5_path.exists():
            action_stats_path = hdf5_path.parent / "action_stats.json"
        if prop_stats_path is None and hdf5_path.exists():
            prop_stats_path = hdf5_path.parent / "prop_stats.json"
        if pcd_stats_path is None and hdf5_path.exists():
            pcd_stats_path = hdf5_path.parent / "pcd_stats.json"
        
        # Copy stats files to run directory
        stats_files_copied = []
        for stats_name, stats_path in [
            ("action_stats.json", action_stats_path),
            ("prop_stats.json", prop_stats_path),
            ("pcd_stats.json", pcd_stats_path),
        ]:
            if stats_path is not None:
                stats_path = Path(stats_path)
                if stats_path.exists():
                    dest_path = run_dir / stats_name
                    shutil.copy2(stats_path, dest_path)
                    stats_files_copied.append(stats_name)
                    print(f"  ✓ Copied {stats_name} to {dest_path}")
                else:
                    print(f"  ⚠ Stats file not found: {stats_path}")
        
        if stats_files_copied:
            print(f"\nStats files saved to: {run_dir}")
            print(f"  Files: {', '.join(stats_files_copied)}")
        print()
    
    # Create module and data module
    print("Creating module and data module...")
    use_dummy = not (use_pcd or use_rgb)
    module = create_module_from_config(config, use_rgb=use_rgb)
    data_module = create_data_module_from_config(config, use_dummy=use_dummy, use_pcd=use_pcd, use_rgb=use_rgb)
    
    encoder_type = "ResNet18" if use_rgb else ("PointNet" if use_pcd else "Dummy")
    print(f"Encoder type: {encoder_type}")
    print(f"Policy parameters: {sum(p.numel() for p in module.policy.parameters()):,}")
    print()
    
    # Create loggers
    loggers = [
        TensorBoardLogger(run_dir, name="tb", version=""),
        CSVLogger(run_dir, name="logs", version=""),
    ]
    
    if config['use_wandb']:
        try:
            # wandb_name은 이미 run_dir 생성시 정의됨 (위에서)
            wandb_logger = WandbLogger(
                project=config['wandb_project'],
                name=wandb_name,
                save_dir=run_dir,
                log_model=False,  # 체크포인트는 로깅하지 않음
                config=config,  # 전체 config를 wandb에 로깅
                tags=[config.get('task', 'brs_policy'), f"seed_{config['seed']}", encoder_type],  # 태그 추가
                notes=f"Training BRS policy ({encoder_type}) with {config['num_latest_obs']} obs window, {config['action_prediction_horizon']} action horizon",
            )
            loggers.append(wandb_logger)
            
            # 추가 메타데이터 로깅
            wandb_logger.experiment.config.update({
                "model_params": sum(p.numel() for p in module.policy.parameters()),
                "trainable_params": sum(p.numel() for p in module.policy.parameters() if p.requires_grad),
            })
            
            print(f"Wandb enabled: {config['wandb_project']}/{wandb_name}")
            print(f"  - Full config and model architecture logged")
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / "ckpt",
            save_on_train_epoch_end=True,
            filename="epoch{epoch}-train_loss{train/loss:.5f}",
            save_top_k=10,  # Save only top 10 checkpoints with lowest train loss
            save_last=True,
            monitor="train/loss",  # Changed from train/loss_epoch
            mode="min",
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=run_dir / "ckpt",
            filename="epoch{epoch}-val_l1{val/l1:.5f}",
            save_top_k=1,  # Save only the best validation checkpoint
            monitor="val/l1",
            mode="min",
            auto_insert_metric_name=False,
        ),
    ]
    
    # Add LearningRateMonitor callback for better logging
    callbacks.append(LearningRateMonitor(logging_interval='step'))
    
    # Add rollout evaluation callback if enabled
    if config.get('rollout_eval', False) and use_pcd:
        rollout_callback = RolloutEvaluationCallback(
            eval_interval=config.get('rollout_eval_interval', 5),
            num_eval_episodes=config.get('num_eval_episodes', 3),
            log_video=config.get('log_eval_video', True),
            max_episode_steps=config.get('max_episode_steps', 500),
            env_name=config.get('env_name', 'SaucepanToHob'),
            cameras=config.get('cameras', ['head', 'left_wrist', 'right_wrist']),
            video_save_dir=str(run_dir / "eval_videos"),  # Save directly under run directory
        )
        callbacks.append(rollout_callback)
        print(f"✓ Rollout evaluation enabled:")
        print(f"  - Interval: every {config.get('rollout_eval_interval', 5)} epochs")
        print(f"  - Episodes: {config.get('num_eval_episodes', 3)}")
        print(f"  - Log video: {config.get('log_eval_video', True)}")
        print(f"  - Video directory: {run_dir / 'eval_videos'}")
    
    # Create trainer
    # Note: Using 32-bit precision instead of 16-mixed to avoid NaN in diffusion loss
    # Diffusion models can be numerically unstable with half precision due to:
    # 1. Very small noise values that underflow in float16
    # 2. MSE loss sum over action dims that can overflow in float16
    # 3. Low learning rates at the end of cosine schedule
    precision_setting = config.get('precision', '32-true')
    trainer = pl.Trainer(
        default_root_dir=run_dir,
        accelerator="gpu" if config['gpus'] > 0 and torch.cuda.is_available() else "cpu",
        devices=config['gpus'] if config['gpus'] > 0 and torch.cuda.is_available() else 1,
        max_epochs=config.get('max_epochs', 999999999),  # Use config value or run indefinitely
        check_val_every_n_epoch=config['eval_interval'],
        gradient_clip_val=config['gradient_clip_val'],
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=config.get('log_every_n_steps', 10),  # 설정 가능하게 변경
        precision=precision_setting,  # Default to 32-bit for stability, configurable
        profiler='simple',  # Enable profiler to identify bottlenecks
    )
    
    # Store config in trainer for callbacks
    trainer.config = trainer_config
    
    # Train
    print("Starting training...")
    print(f"Validation every {config['eval_interval']} epochs")
    print("Press Ctrl+C to stop training")
    print()
    
    try:
        trainer.fit(module, data_module)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    print("="*80)
    print("Training completed!")
    print(f"Checkpoints saved to: {run_dir / 'ckpt'}")
    print("="*80)
    
    return module, trainer


# ============================================================================
# CLI
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train BRS Policy with PyTorch Lightning')
    parser.add_argument(
        '--config',
        type=str,
        default='../cfgs/brs_config.yaml',
        help='Path to config file'
    )
    parser.add_argument('--use-pcd', action='store_true', help='Use PCD dataset with PointNet encoder')
    parser.add_argument('--use-rgb', action='store_true', help='Use RGB dataset with ResNet18 encoder')
    parser.add_argument('--hdf5-path', type=str, default=None, help='Path to HDF5 file (overrides config)')
    parser.add_argument('--pcd-root', type=str, default=None, help='Path to PCD directory (overrides config)')
    parser.add_argument('--gpus', type=int, default=None, help='Number of GPUs')
    parser.add_argument('--bs', type=int, default=None, help='Batch size')
    parser.add_argument('--vbs', type=int, default=None, help='Validation batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--wandb-name', type=str, default=None, help='Wandb run name (overrides config)')
    parser.add_argument('--dataloader-num-workers', type=int, default=None, help='Number of dataloader workers')
    parser.add_argument('--dataloader-prefetch-factor', type=int, default=None, help='Prefetch factor for dataloader')
    parser.add_argument('--dataloader-persistent-workers', action='store_true', help='Use persistent workers')
    parser.add_argument('--preload-hdf5', action='store_true', help='Preload all HDF5 data into RAM (~100MB)')
    parser.add_argument('--preload-pcd', action='store_true', help='Preload all PCD data into RAM (~3GB)')
    parser.add_argument('--preload-data', action='store_true', help='Preload all data into RAM (for RGB mode)')
    
    args = parser.parse_args()
    
    # Prepare overrides
    overrides = {}
    if args.hdf5_path is not None:
        overrides['hdf5_path'] = args.hdf5_path
    if args.pcd_root is not None:
        overrides['pcd_root'] = args.pcd_root
    if args.gpus is not None:
        overrides['gpus'] = args.gpus
    if args.bs is not None:
        overrides['bs'] = args.bs
    if args.vbs is not None:
        overrides['vbs'] = args.vbs
    if args.lr is not None:
        overrides['lr'] = args.lr
    if args.seed is not None:
        overrides['seed'] = args.seed
    if args.no_wandb:
        overrides['use_wandb'] = False
    if args.wandb_name is not None:
        overrides['wandb_name'] = args.wandb_name
    if args.dataloader_num_workers is not None:
        overrides['dataloader_num_workers'] = args.dataloader_num_workers
    if args.dataloader_prefetch_factor is not None:
        overrides['prefetch_factor'] = args.dataloader_prefetch_factor
    if args.dataloader_persistent_workers:
        overrides['persistent_workers'] = True
    if args.preload_hdf5:
        overrides['preload_hdf5'] = True
    if args.preload_pcd:
        overrides['preload_pcd'] = True
    if args.preload_data:
        overrides['preload_data'] = True
    
    # Get config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        # If relative path, resolve from current working directory
        config_path = Path.cwd() / config_path
    
    # Train
    train(str(config_path), use_pcd=args.use_pcd, use_rgb=args.use_rgb, **overrides)
