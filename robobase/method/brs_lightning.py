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
from brs_algo.learning.policy import WBVIMAPolicy
from brs_algo.learning.module import DiffusionModule
from brs_algo.learning.data import ActionSeqChunkDataModule
from diffusers.schedulers.scheduling_ddim import DDIMScheduler


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
                "torso": np.random.randn(self.num_latest_obs, 4).astype(np.float32),
                "left_arm": np.random.randn(self.num_latest_obs, 6).astype(np.float32),
                "left_gripper": np.random.randn(self.num_latest_obs, 1).astype(np.float32),
                "right_arm": np.random.randn(self.num_latest_obs, 6).astype(np.float32),
                "right_gripper": np.random.randn(self.num_latest_obs, 1).astype(np.float32),
            },
            "pointcloud": {
                "xyz": pcd_data[..., :3],  # (num_latest_obs, n_points, 3)
                "rgb": pcd_data[..., 3:],  # (num_latest_obs, n_points, 3)
            }
        }
        
        # Generate dummy actions matching action_keys
        # Actions have shape: (num_latest_obs, action_prediction_horizon, dim)
        action_chunks = {
            "mobile_base": np.random.randn(self.num_latest_obs, self.action_prediction_horizon, 3).astype(np.float32),
            "torso": np.random.randn(self.num_latest_obs, self.action_prediction_horizon, 4).astype(np.float32),
            "left_arm": np.random.randn(self.num_latest_obs, self.action_prediction_horizon, 6).astype(np.float32),
            "left_gripper": np.random.randn(self.num_latest_obs, self.action_prediction_horizon, 1).astype(np.float32),
            "right_arm": np.random.randn(self.num_latest_obs, self.action_prediction_horizon, 6).astype(np.float32),
            "right_gripper": np.random.randn(self.num_latest_obs, self.action_prediction_horizon, 1).astype(np.float32),
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
        "left_arm": torch.from_numpy(np.stack([b["action_chunks"]["left_arm"] for b in batch])),
        "left_gripper": torch.from_numpy(np.stack([b["action_chunks"]["left_gripper"] for b in batch])),
        "right_arm": torch.from_numpy(np.stack([b["action_chunks"]["right_arm"] for b in batch])),
        "right_gripper": torch.from_numpy(np.stack([b["action_chunks"]["right_gripper"] for b in batch])),
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
            persistent_workers=self.dataloader_num_workers > 0
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.dataloader_num_workers,
            collate_fn=brs_collate_fn,
            persistent_workers=self.dataloader_num_workers > 0
        )


# ============================================================================
# Config Loading
# ============================================================================


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML config file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
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


def create_module_from_config(config: Dict[str, Any]) -> DiffusionModule:
    """Create DiffusionModule from config"""
    policy = create_policy_from_config(config)
    
    module = DiffusionModule(
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



def create_data_module_from_config(config: Dict[str, Any], use_dummy: bool = True):
    """
    Create DataModule from config
    
    Args:
        config: Config dict
        use_dummy: If True, use DummyDataModule. If False, use ActionSeqChunkDataModule (needs real data)
    """
    if use_dummy:
        data_module = DummyDataModule(
            num_latest_obs=config['num_latest_obs'],
            action_prediction_horizon=config['action_prediction_horizon'],
            batch_size=config['bs'],
            prop_dim=config['prop_dim'],
            pcd_downsample_points=config['pcd_downsample_points'],
            val_batch_size=config['vbs'],
            val_split_ratio=config['val_split_ratio'],
            dataloader_num_workers=config['dataloader_num_workers'],
            seed=config['seed'] if config['seed'] > 0 else None,
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

def train(config_path: str, **overrides):
    """
    Main training function following brs-algo structure
    
    Args:
        config_path: Path to config YAML file
        **overrides: Override config values
    """
    # Load config
    config = load_config(config_path)
    
    # Apply overrides
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    
    # Set seed
    if config['seed'] > 0:
        pl.seed_everything(config['seed'], workers=True)
    
    # Print config
    print("="*80)
    print("BRS Policy Training (PyTorch Lightning)")
    print("="*80)
    print(f"Run name: {config['run_name']}")
    print(f"Config: {config_path}")
    print(f"Batch size: {config['bs']} (val: {config['vbs']})")
    print(f"Learning rate: {config['lr']}")
    print(f"Weight decay: {config['wd']}")
    print(f"Num latest obs: {config['num_latest_obs']}")
    print(f"Action prediction horizon: {config['action_prediction_horizon']}")
    print()
    
    # Create run directory
    run_dir = Path("runs") / config['run_name']
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(run_dir / "config.yaml", 'w') as f:
        yaml.dump(config, f)
    
    # Create module and data module
    print("Creating module and data module...")
    module = create_module_from_config(config)
    data_module = create_data_module_from_config(config)
    
    print(f"Policy parameters: {sum(p.numel() for p in module.policy.parameters()):,}")
    print()
    
    # Create loggers
    loggers = [
        TensorBoardLogger(run_dir, name="tb", version=""),
        CSVLogger(run_dir, name="logs", version=""),
    ]
    
    if config['use_wandb']:
        try:
            wandb_logger = WandbLogger(
                project=config['wandb_project'],
                name=config['wandb_run_name'],
                save_dir=run_dir,
            )
            loggers.append(wandb_logger)
            print(f"Wandb enabled: {config['wandb_project']}/{config['wandb_run_name']}")
        except Exception as e:
            print(f"Warning: Could not initialize wandb: {e}")
    
    # Create callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=run_dir / "ckpt",
            save_on_train_epoch_end=True,
            filename="epoch{epoch}-train_loss{train/loss:.5f}",
            save_top_k=100,
            save_last=True,
            monitor="train/loss",  # Changed from train/loss_epoch
            mode="min",
            auto_insert_metric_name=False,
        ),
        ModelCheckpoint(
            dirpath=run_dir / "ckpt",
            filename="epoch{epoch}-val_l1{val/l1:.5f}",
            save_top_k=-1,
            monitor="val/l1",
            mode="min",
            auto_insert_metric_name=False,
        ),
    ]
    
    # Create trainer
    trainer = pl.Trainer(
        default_root_dir=run_dir,
        accelerator="gpu" if config['gpus'] > 0 and torch.cuda.is_available() else "cpu",
        devices=config['gpus'] if config['gpus'] > 0 and torch.cuda.is_available() else 1,
        max_epochs=999999999,  # Run indefinitely
        check_val_every_n_epoch=config['eval_interval'],
        gradient_clip_val=config['gradient_clip_val'],
        logger=loggers,
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=10,
    )
    
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
    parser.add_argument('--gpus', type=int, default=None, help='Number of GPUs')
    parser.add_argument('--bs', type=int, default=None, help='Batch size')
    parser.add_argument('--vbs', type=int, default=None, help='Validation batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb')
    
    args = parser.parse_args()
    
    # Prepare overrides
    overrides = {}
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
    
    # Get config path
    config_path = Path(__file__).parent / args.config
    
    # Train
    train(str(config_path), **overrides)
