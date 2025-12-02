#!/usr/bin/env python3
"""
ACT Policy Training Script for SaucepanToHob HDF5 Dataset.

Standalone training script that directly loads from demos.hdf5.
Uses RGB images and 16D proprioception for behavior cloning.

Action Space (16D with torso=False):
    [0:4]   Floating base delta [x, y, z, rz]
    [4:9]   Left arm (5D)
    [9:14]  Right arm (5D)
    [14]    Left gripper
    [15]    Right gripper

Proprioception (16D):
    [0:4]   Floating base position [x, y, z, rz] (reconstructed from action integration)
    [4:9]   Left arm (5D)
    [9:14]  Right arm (5D)
    [14]    Left gripper
    [15]    Right gripper

Note: HDF5 has native 16D actions. Proprioception is reconstructed from qpos.

Results are saved to: runs/${run_name}/
    - config.json
    - train.log
    - checkpoints/best_val_loss.pt
    - checkpoints/best_success.pt
    - checkpoints/checkpoint_epoch_NNNN.pt
    - eval_videos/epoch_NNNN_ep_N.mp4

Usage:
    python train_act_hdf5.py --run-name my_experiment
    python train_act_hdf5.py --run-name exp1 --wandb
    python train_act_hdf5.py --run-name exp1 --eval-every 50 --eval-episodes 3
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add robobase to path
sys.path.insert(0, str(Path(__file__).parent))

from robobase.method.act_hdf5_dataset import ACTHdf5Dataset, ACTHdf5DataModule
from robobase.models.multi_view_transformer import MultiViewTransformerEncoderDecoderACT
from robobase.models.act.backbone import build_backbone


class ImageEncoderACT(nn.Module):
    """
    Image Encoder for ACT model.
    Simplified version for standalone training.
    """
    
    VISUAL_OBS_MEAN = [0.485, 0.456, 0.406]
    VISUAL_OBS_STD = [0.229, 0.224, 0.225]
    
    def __init__(
        self,
        num_views: int = 1,
        frame_stack: int = 1,
        hidden_dim: int = 512,
        backbone: str = "resnet18",
    ):
        super().__init__()
        self.num_views = num_views
        self.frame_stack = frame_stack
        self.hidden_dim = hidden_dim
        
        self.backbone = build_backbone(
            hidden_dim=hidden_dim,
            position_embedding="sine",
            lr_backbone=1e-5,
            masks=False,
            backbone=backbone,
            dilation=False,
        )
        
        for param in self.backbone.parameters():
            param.requires_grad = True
        
        self.input_proj = nn.Conv2d(
            self.backbone.num_channels, hidden_dim, kernel_size=1
        )
    
    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, V, FS, C, H, W) or (B, V*FS, C, H, W) tensor
        Returns:
            feat: (B, hidden_dim, H', W')
            pos: (B, hidden_dim, H', W')
        """
        B = x.shape[0]
        
        # Handle different input shapes
        if len(x.shape) == 6:
            # (B, V, FS, C, H, W) -> (B*V*FS, C, H, W)
            V, FS, C, H, W = x.shape[1:]
            x = x.view(B * V * FS, C, H, W)
        else:
            # (B, V*FS, C, H, W) -> (B*V*FS, C, H, W)
            VFS, C, H, W = x.shape[1:]
            x = x.view(B * VFS, C, H, W)
            V = self.num_views
            FS = VFS // V
        
        # Backbone forward
        feat, pos = self.backbone(x)
        feat = self.input_proj(feat[0])
        pos = pos[0]
        
        # Reshape and average over views/frames to keep hidden_dim
        # (B*V*FS, D, H', W') -> (B, V*FS, D, H', W') -> (B, D, H', W')
        _, D, H_out, W_out = feat.shape
        feat = feat.view(B, V * FS, D, H_out, W_out)
        feat = feat.mean(dim=1)  # Average over views and frames
        
        return feat, pos


class ACTPolicySimple(nn.Module):
    """
    Simplified ACT Policy for standalone training.
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        num_views: int = 1,
        frame_stack: int = 1,
        hidden_dim: int = 512,
        num_queries: int = 16,
        enc_layers: int = 4,
        dec_layers: int = 1,
        nheads: int = 8,
        dim_feedforward: int = 3200,
        backbone: str = "resnet18",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        
        # Image encoder
        self.encoder = ImageEncoderACT(
            num_views=num_views,
            frame_stack=frame_stack,
            hidden_dim=hidden_dim,
            backbone=backbone,
        )
        
        # Transformer decoder (ACT style)
        self.transformer = MultiViewTransformerEncoderDecoderACT(
            input_shape=(hidden_dim, 7, 7),  # Approximate for resnet18 with 224x224 input
            hidden_dim=hidden_dim,
            enc_layers=enc_layers,
            dec_layers=dec_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            nheads=nheads,
            num_queries=num_queries,
            pre_norm=False,
            state_dim=state_dim,
            action_dim=action_dim,
            use_lang_cond=False,
        )
        
        # Image normalization
        self.register_buffer(
            'img_mean',
            torch.tensor(ImageEncoderACT.VISUAL_OBS_MEAN).view(1, 1, 3, 1, 1)
        )
        self.register_buffer(
            'img_std',
            torch.tensor(ImageEncoderACT.VISUAL_OBS_STD).view(1, 1, 3, 1, 1)
        )
    
    def normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images to ImageNet statistics."""
        # images: (B, FS, C, H, W)
        images = images / 255.0
        images = (images - self.img_mean) / self.img_std
        return images
    
    def forward(
        self,
        images: torch.Tensor,
        qpos: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
        is_pad: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass.
        
        Args:
            images: (B, FS, C, H, W) RGB images
            qpos: (B, state_dim) proprioceptive state
            actions: (B, num_queries, action_dim) target actions (for training)
            is_pad: (B, num_queries) padding mask
        
        Returns:
            If actions provided: loss dict
            Else: predicted actions (B, num_queries, action_dim)
        """
        B = images.shape[0]
        
        # Normalize images
        images = self.normalize_images(images)
        
        # Add view dimension if needed: (B, FS, C, H, W) -> (B, 1, FS, C, H, W)
        if len(images.shape) == 5:
            images = images.unsqueeze(1)
        
        # Reshape for encoder: (B, V, FS, C, H, W) -> (B, V*FS, C, H, W)
        B, V, FS, C, H, W = images.shape
        images = images.view(B, V * FS, C, H, W)
        
        # Encode images
        feat, pos = self.encoder(images)
        
        # Prepare for transformer
        src = (feat, pos)
        
        # Transformer forward
        if actions is not None:
            # Training mode
            output = self.transformer(src, qpos, actions=actions, is_pad=is_pad)
            loss, loss_dict = self.transformer.calculate_loss(
                output, actions=actions, is_pad=is_pad
            )
            return loss_dict
        else:
            # Inference mode
            output = self.transformer(src, qpos, actions=None, is_pad=None)
            return output
    
    @torch.no_grad()
    def get_action(self, images: torch.Tensor, qpos: torch.Tensor) -> torch.Tensor:
        """
        Get action prediction for inference.
        
        Args:
            images: (B, FS, C, H, W) RGB images
            qpos: (B, state_dim) proprioceptive state
        
        Returns:
            a_hat: (B, num_queries, action_dim) predicted action sequence
        """
        self.eval()
        output = self.forward(images, qpos, actions=None, is_pad=None)
        # output is (a_hat, is_pad_hat, [mu, logvar])
        a_hat = output[0]  # (B, num_queries, action_dim)
        return a_hat


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    gradient_clip: Optional[float] = None,
) -> Dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_l1_loss = 0.0
    total_kl_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        # Move to device
        images = batch['rgb_head'].to(device)  # (B, FS, C, H, W)
        qpos = batch['low_dim_state'].to(device)  # (B, state_dim)
        actions = batch['action'].to(device)  # (B, num_queries, action_dim)
        is_pad = batch['is_pad'].to(device)  # (B, num_queries)
        
        # Forward
        optimizer.zero_grad()
        loss_dict = model(images, qpos, actions=actions, is_pad=is_pad)
        
        loss = loss_dict['loss']
        
        # Backward
        loss.backward()
        
        if gradient_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss_dict['loss'].item()
        if 'l1' in loss_dict:
            total_l1_loss += loss_dict['l1'].item()
        if 'kl' in loss_dict:
            total_kl_loss += loss_dict['kl'].item()
        n_batches += 1
        
        pbar.set_postfix({
            'loss': f"{total_loss/n_batches:.4f}",
            'l1': f"{total_l1_loss/n_batches:.4f}",
        })
    
    return {
        'loss': total_loss / n_batches,
        'l1_loss': total_l1_loss / n_batches,
        'kl_loss': total_kl_loss / n_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_l1_loss = 0.0
    n_batches = 0
    
    for batch in tqdm(dataloader, desc="Validation"):
        images = batch['rgb_head'].to(device)
        qpos = batch['low_dim_state'].to(device)
        actions = batch['action'].to(device)
        is_pad = batch['is_pad'].to(device)
        
        loss_dict = model(images, qpos, actions=actions, is_pad=is_pad)
        
        total_loss += loss_dict['loss'].item()
        if 'l1' in loss_dict:
            total_l1_loss += loss_dict['l1'].item()
        n_batches += 1
    
    return {
        'val_loss': total_loss / n_batches,
        'val_l1_loss': total_l1_loss / n_batches,
    }


def create_eval_env():
    """Create evaluation environment with same config as training demos."""
    try:
        from bigym.action_modes import JointPositionActionMode
        from bigym.robots.configs.h1 import H1FineManipulation, PelvisDof
        from bigym.envs.pick_and_place import SaucepanToHob
        from bigym.utils.observation_config import ObservationConfig, CameraConfig
        
        floating_dofs = [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
        
        # Note: h1.py has torso: False, so action_dim = 16 (not 17)
        # floating_base(4) + left_arm(5) + right_arm(5) + grippers(2) = 16
        env = SaucepanToHob(
            robot_cls=H1FineManipulation,
            action_mode=JointPositionActionMode(
                floating_base=True,
                floating_dofs=floating_dofs,
                absolute=True,
            ),
            observation_config=ObservationConfig(
                cameras=[
                    CameraConfig(name="head", rgb=True, depth=False, resolution=(224, 224)),
                ],
            ),
            render_mode="rgb_array",
        )
        
        # Verify action dimension
        expected_action_dim = 16  # h1.py torso: False
        actual_action_dim = env.action_space.shape[0]
        if actual_action_dim != expected_action_dim:
            logging.warning(f"Action dim mismatch: expected {expected_action_dim}, got {actual_action_dim}")
            logging.warning(f"Make sure bigym/robots/configs/h1.py has 'torso: False'")
        
        return env
    except Exception as e:
        logging.warning(f"Failed to create eval environment: {e}")
        import traceback
        traceback.print_exc()
        return None


def unnormalize_action(action: np.ndarray, action_stats: Dict) -> np.ndarray:
    """Unnormalize action from [-1, 1] to original range.
    
    Handles preprocessed stats format with 'min'/'max' arrays.
    """
    min_val = action_stats['min']
    max_val = action_stats['max']
    
    if not isinstance(min_val, np.ndarray):
        min_val = np.array(min_val)
    if not isinstance(max_val, np.ndarray):
        max_val = np.array(max_val)
    
    # From [-1, 1] to [0, 1]
    action_normalized = (action + 1) / 2
    # From [0, 1] to [min, max]
    action_unnorm = action_normalized * (max_val - min_val) + min_val
    return action_unnorm.astype(np.float32)


def load_action_stats_for_eval(stats_path: str) -> Dict:
    """Load action stats and convert to flat min/max arrays for evaluation.
    
    Supports multiple formats:
    1. Flat format: {min: [16], max: [16]}
    2. 'full' key format: {full: {min: [16], max: [16]}}
    3. BRS format: {mobile_base: {...}, torso: {...}, arms: {...}}
    """
    with open(stats_path, 'r') as f:
        raw_stats = json.load(f)
    
    # Format 1: Flat format
    if 'min' in raw_stats and isinstance(raw_stats['min'], list) and len(raw_stats['min']) == 16:
        return {
            'min': np.array(raw_stats['min'], dtype=np.float32),
            'max': np.array(raw_stats['max'], dtype=np.float32),
        }
    
    # Format 2: 'full' key format
    if 'full' in raw_stats:
        return {
            'min': np.array(raw_stats['full']['min'], dtype=np.float32),
            'max': np.array(raw_stats['full']['max'], dtype=np.float32),
        }
    
    # Format 3: BRS format -> convert to BigYM 16D
    if 'mobile_base' in raw_stats and 'torso' in raw_stats:
        mb = raw_stats['mobile_base']
        torso = raw_stats['torso']
        arms = raw_stats['arms']
        
        min_list = []
        max_list = []
        
        # floating_base: [dx, dy, dz, drz]
        min_list.extend([mb['min'][0], mb['min'][1]])
        min_list.append(torso['min'] if not isinstance(torso['min'], list) else torso['min'][0])
        min_list.append(mb['min'][2])
        
        max_list.extend([mb['max'][0], mb['max'][1]])
        max_list.append(torso['max'] if not isinstance(torso['max'], list) else torso['max'][0])
        max_list.append(mb['max'][2])
        
        # arms: BRS has [left_arm(5), left_grip(1), right_arm(5), right_grip(1)]
        # BigYM wants [left_arm(5), right_arm(5), grippers(2)]
        min_list.extend(arms['min'][0:5])   # left_arm
        min_list.extend(arms['min'][6:11])  # right_arm
        min_list.append(arms['min'][5])     # left_gripper
        min_list.append(arms['min'][11])    # right_gripper
        
        max_list.extend(arms['max'][0:5])
        max_list.extend(arms['max'][6:11])
        max_list.append(arms['max'][5])
        max_list.append(arms['max'][11])
        
        return {
            'min': np.array(min_list, dtype=np.float32),
            'max': np.array(max_list, dtype=np.float32),
        }
    
    # Format 4: Legacy ACT format
    if 'floating_base' in raw_stats:
        min_list = list(raw_stats['floating_base']['min'])
        max_list = list(raw_stats['floating_base']['max'])
        
        if 'arms' in raw_stats:
            min_list.extend(raw_stats['arms']['min'])
            max_list.extend(raw_stats['arms']['max'])
        
        return {
            'min': np.array(min_list, dtype=np.float32),
            'max': np.array(max_list, dtype=np.float32),
        }
    
    raise ValueError(f"Unknown action stats format, keys: {list(raw_stats.keys())}")


def clip_action_to_env_bounds(action: np.ndarray, env) -> np.ndarray:
    """Clip action to environment action space bounds."""
    return np.clip(action, env.action_space.low, env.action_space.high)


def convert_16d_to_17d(action_16d: np.ndarray) -> np.ndarray:
    """
    Convert 16D demo action to 17D env action (if torso is enabled).
    Insert torso=0 at index 4 (torso is not used in 16D).
    
    NOTE: If h1.py has torso: False, this function is NOT needed.
    The environment will directly accept 16D actions.
    
    16D: [fb(4), left_arm(5), right_arm(5), left_gripper, right_gripper]
    17D: [fb(4), torso(1), left_arm(5), right_arm(5), left_gripper, right_gripper]
    """
    action_17d = np.zeros(17, dtype=np.float32)
    action_17d[0:4] = action_16d[0:4]    # floating base
    action_17d[4] = 0.0                   # torso (always 0)
    action_17d[5:17] = action_16d[4:16]  # arms + grippers
    return action_17d


def save_video(frames: List[np.ndarray], path: Path, fps: int = 30):
    """Save frames as MP4 video with H.264 codec for better compatibility."""
    if not frames:
        return False
    
    # Try imageio first (better H.264 support)
    try:
        import imageio
        
        # Use imageio-ffmpeg for H.264 encoding
        writer = imageio.get_writer(
            str(path),
            fps=fps,
            codec='libx264',
            quality=8,  # 0-10, higher is better
            pixelformat='yuv420p',  # Required for compatibility
            output_params=['-preset', 'fast'],
        )
        
        for frame in frames:
            # Ensure RGB format
            if len(frame.shape) == 2:
                frame = np.stack([frame] * 3, axis=-1)
            writer.append_data(frame)
        
        writer.close()
        return True
    except Exception as e:
        logging.warning(f"imageio failed: {e}, trying cv2...")
    
    # Fallback to cv2
    try:
        import cv2
        
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(path), fourcc, fps, (w, h))
        
        for frame in frames:
            if frame.shape[-1] == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            out.write(frame)
        
        out.release()
        return True
    except Exception as e:
        logging.warning(f"Failed to save video: {e}")
        return False


# Environment runs at 500Hz, but demo data is stored at 50Hz (decimation=10)
# Floating base actions in demo are SUM of 10 x 500Hz deltas
# To replay correctly: divide floating base by 10, execute 10 env steps per policy action
ENV_DECIMATION = 10


@torch.no_grad()
def evaluate_in_env(
    model: nn.Module,
    device: torch.device,
    action_stats: Dict,
    num_episodes: int = 3,
    max_steps: int = 1000,
    video_dir: Optional[Path] = None,
    epoch: int = 0,
    temporal_ensemble_k: int = 4,
    decimation: int = ENV_DECIMATION,
) -> Tuple[Dict, List]:
    """
    Evaluate policy in BigYM environment and optionally save videos.
    
    IMPORTANT: Environment runs at 500Hz but policy outputs actions at 50Hz.
    Floating base actions are deltas accumulated over 10 timesteps in the demo.
    We must divide floating base by decimation and execute decimation steps.
    
    Args:
        model: ACT policy model
        device: torch device
        action_stats: Action statistics for unnormalization
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode (at 50Hz, so actual env steps = max_steps * decimation)
        video_dir: Directory to save videos (None to skip)
        epoch: Current epoch number (for video naming)
        temporal_ensemble_k: Number of actions to use for temporal ensemble
        decimation: Number of env steps per policy action (default: 10 for 500Hz/50Hz)
    
    Returns:
        metrics: Dict with eval_success_rate, eval_avg_reward, etc.
        video_frames_list: List of frame lists for each episode
    """
    env = create_eval_env()
    if env is None:
        logging.warning("Could not create eval environment, skipping evaluation")
        return {'eval_success_rate': 0.0, 'eval_avg_reward': 0.0}, []
    
    model.eval()
    
    successes = []
    total_rewards = []
    video_frames_list = []
    
    # QPOS indices for arms (matching dataset extraction)
    # H1 robot qpos structure (30D):
    # qpos[0-3]: left_shoulder_pitch, left_shoulder_roll, left_shoulder_yaw, left_elbow
    # qpos[4-11]: left_gripper joints (8)
    # qpos[12]: left_wrist
    # qpos[13-16]: right_shoulder_pitch, right_shoulder_roll, right_shoulder_yaw, right_elbow
    # qpos[17-24]: right_gripper joints (8)
    # qpos[25]: right_wrist
    # qpos[26-29]: pelvis_x, pelvis_y, pelvis_z, pelvis_rz
    QPOS_LEFT_ARM = [0, 1, 2, 3, 12]   # 5D: shoulder_pitch, roll, yaw, elbow, wrist
    QPOS_RIGHT_ARM = [13, 14, 15, 16, 25]  # 5D: shoulder_pitch, roll, yaw, elbow, wrist
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        
        frames = []
        episode_reward = 0.0
        done = False
        
        # Action queue for temporal ensemble / action chunking
        action_queue = []
        
        for step in range(max_steps):
            # Render frame for video
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # Prepare RGB observation: already (C, H, W) from env
            rgb = obs['rgb_head']
            if len(rgb.shape) == 3 and rgb.shape[-1] == 3:
                rgb = rgb.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            # Get proprioception (16D) matching dataset format:
            # [floating_base(4), left_arm(5), right_arm(5), grippers(2)]
            qpos = obs['proprioception'][:30]  # qpos only (first 30)
            floating_base = obs['proprioception_floating_base']  # (4,) [x, y, z, rz]
            grippers = obs['proprioception_grippers']  # (2,) [left, right]
            
            left_arm = qpos[QPOS_LEFT_ARM]   # (5,)
            right_arm = qpos[QPOS_RIGHT_ARM]  # (5,)
            
            prop_16d = np.concatenate([floating_base, left_arm, right_arm, grippers])
            
            # Get new action chunk if queue is empty
            if len(action_queue) == 0:
                rgb_tensor = torch.from_numpy(rgb).unsqueeze(0).unsqueeze(0).float().to(device)
                prop_tensor = torch.from_numpy(prop_16d).unsqueeze(0).float().to(device)
                
                action_pred = model.get_action(rgb_tensor, prop_tensor)
                action_chunk = action_pred[0].cpu().numpy()  # (num_queries, action_dim)
                
                # Use temporal ensemble: only use first K actions, then get new chunk
                action_queue = list(action_chunk[:temporal_ensemble_k])
            
            # Get next action from queue
            action_16d_norm = action_queue.pop(0)
            
            # Unnormalize action
            action_16d = unnormalize_action(action_16d_norm, action_stats)
            
            # CRITICAL: Apply decimation for 500Hz/50Hz frequency mismatch
            # Policy outputs 50Hz actions, but env runs at 500Hz
            # Floating base deltas in demo are SUM of 10 x 500Hz deltas
            # So we divide floating base by decimation and execute decimation env steps
            action_500hz = action_16d.copy()
            action_500hz[0:4] = action_16d[0:4] / decimation  # Divide floating base delta
            
            # Execute decimation env steps
            step_reward = 0.0
            for _ in range(decimation):
                # Clip to environment bounds
                action = clip_action_to_env_bounds(action_500hz, env)
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                step_reward += reward
                done = terminated or truncated
                
                if done:
                    break
            
            episode_reward += step_reward
            
            if done:
                break
        
        # Check success
        success = info.get('success', False) or info.get('is_success', False)
        successes.append(float(success))
        total_rewards.append(episode_reward)
        
        if frames:
            video_frames_list.append(frames)
        
        logging.info(f"  Episode {ep+1}: steps={step+1}, reward={episode_reward:.2f}, success={success}")
    
    env.close()
    
    # Save videos
    if video_dir is not None and video_frames_list:
        video_dir.mkdir(parents=True, exist_ok=True)
        for ep_idx, frames in enumerate(video_frames_list):
            video_path = video_dir / f"epoch_{epoch:04d}_ep_{ep_idx}.mp4"
            if save_video(frames, video_path):
                logging.info(f"  Video saved: {video_path}")
    
    metrics = {
        'eval_success_rate': np.mean(successes),
        'eval_avg_reward': np.mean(total_rewards),
        'eval_episodes': num_episodes,
    }
    
    return metrics, video_frames_list


def main():
    parser = argparse.ArgumentParser(description='Train ACT Policy on HDF5 Dataset')
    
    # Run name
    parser.add_argument('--run-name', type=str, default=None,
                       help='Run name (default: act_YYYYMMDD_HHMMSS)')
    
    # Data
    parser.add_argument('--hdf5-path', type=str,
                       default='/home/hyunjin/bigym_ws/data/demonstrations/saucepan_to_hob/demos.hdf5',
                       help='Path to HDF5 dataset')
    parser.add_argument('--action-stats-path', type=str, default=None,
                       help='Path to action_stats.json (default: same dir as HDF5)')
    
    # Model
    parser.add_argument('--hidden-dim', type=int, default=512,
                       help='Transformer hidden dimension')
    parser.add_argument('--action-sequence', type=int, default=16,
                       help='Action chunk size (num queries)')
    parser.add_argument('--enc-layers', type=int, default=4,
                       help='Number of encoder layers')
    parser.add_argument('--dec-layers', type=int, default=1,
                       help='Number of decoder layers')
    parser.add_argument('--backbone', type=str, default='resnet18',
                       help='Image backbone')
    
    # Training
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--lr-backbone', type=float, default=1e-5,
                       help='Backbone learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--epochs', type=int, default=300,
                       help='Number of training epochs')
    parser.add_argument('--gradient-clip', type=float, default=10.0,
                       help='Gradient clipping norm')
    parser.add_argument('--val-ratio', type=float, default=0.1,
                       help='Validation split ratio')
    parser.add_argument('--num-workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Image
    parser.add_argument('--image-size', type=int, default=224,
                       help='Image size (square)')
    parser.add_argument('--frame-stack', type=int, default=1,
                       help='Number of frames to stack')
    
    # Data loading optimization
    parser.add_argument('--preload-images', action='store_true',
                       help='Preload all RGB images into memory for faster training (~5GB for 100 demos)')
    parser.add_argument('--preload-all', action='store_true',
                       help='Preload all data (images + proprioception + actions) into memory')
    
    # Checkpointing
    parser.add_argument('--save-every', type=int, default=50,
                       help='Save checkpoint every N epochs')
    
    # Evaluation in environment
    parser.add_argument('--eval-every', type=int, default=0,
                       help='Evaluate in env every N epochs (0 to disable)')
    parser.add_argument('--eval-episodes', type=int, default=3,
                       help='Number of evaluation episodes')
    parser.add_argument('--eval-max-steps', type=int, default=1000,
                       help='Maximum steps per evaluation episode')
    parser.add_argument('--temporal-ensemble-k', type=int, default=4,
                       help='Number of actions for temporal ensemble')
    
    # Wandb
    parser.add_argument('--wandb', action='store_true',
                       help='Enable wandb logging')
    parser.add_argument('--wandb-project', type=str, default='bigym_act',
                       help='Wandb project name')
    parser.add_argument('--wandb-entity', type=str, default=None,
                       help='Wandb entity (team/user)')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Generate run name if not provided
    if args.run_name is None:
        args.run_name = f"act_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Setup paths: runs/${run_name}/
    run_dir = Path(__file__).parent / 'runs' / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = run_dir / 'checkpoints'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    eval_video_dir = run_dir / 'eval_videos'
    eval_video_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = run_dir / 'train.log'
    
    # Setup logging (file + console)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Run name: {args.run_name}")
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA not available, using CPU")
        args.device = 'cpu'
    device = torch.device(args.device)
    logging.info(f"Device: {device}")
    
    # Setup wandb
    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=vars(args),
            dir=str(run_dir),
        )
        logging.info(f"Wandb run: {wandb_run.url}")
    
    # Save config
    config_path = run_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logging.info(f"Config saved to {config_path}")
    
    # Load action stats for evaluation
    action_stats_path = args.action_stats_path
    if action_stats_path is None:
        action_stats_path = str(Path(args.hdf5_path).parent / 'action_stats.json')
    
    action_stats = load_action_stats_for_eval(action_stats_path)
    logging.info(f"Action stats loaded from {action_stats_path}")
    logging.info(f"  min shape: {action_stats['min'].shape}, max shape: {action_stats['max'].shape}")
    
    # Create data module
    data_module = ACTHdf5DataModule(
        hdf5_path=args.hdf5_path,
        action_stats_path=action_stats_path,
        val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=(args.image_size, args.image_size),
        frame_stack=args.frame_stack,
        action_sequence=args.action_sequence,
        normalize_actions=True,
        min_max_margin=0.0,
        preload_images=args.preload_images,
        preload_all=args.preload_all,
    )
    data_module.setup()
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    logging.info(f"Train samples: {len(data_module.train_dataset)}")
    logging.info(f"Val samples: {len(data_module.val_dataset)}")
    
    # Get dimensions from dataset
    sample = data_module.train_dataset[0]
    state_dim = sample['low_dim_state'].shape[0]
    action_dim = sample['action'].shape[1]
    
    logging.info(f"State dim: {state_dim}")
    logging.info(f"Action dim: {action_dim}")
    
    # Create model
    model = ACTPolicySimple(
        state_dim=state_dim,
        action_dim=action_dim,
        num_views=1,
        frame_stack=args.frame_stack,
        hidden_dim=args.hidden_dim,
        num_queries=args.action_sequence,
        enc_layers=args.enc_layers,
        dec_layers=args.dec_layers,
        backbone=args.backbone,
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total parameters: {total_params:,}")
    logging.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer with different LR for backbone
    backbone_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': args.lr},
        {'params': backbone_params, 'lr': args.lr_backbone},
    ], weight_decay=args.weight_decay)
    
    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_success_rate = 0.0
    
    for epoch in range(1, args.epochs + 1):
        logging.info(f"\n{'='*60}")
        logging.info(f"Epoch {epoch}/{args.epochs}")
        logging.info(f"{'='*60}")
        
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, epoch,
            gradient_clip=args.gradient_clip
        )
        
        # Validate
        val_metrics = validate(model, val_loader, device)
        
        # Update LR
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Log
        logging.info(f"Train Loss: {train_metrics['loss']:.4f}, L1: {train_metrics['l1_loss']:.4f}, KL: {train_metrics['kl_loss']:.4f}")
        logging.info(f"Val Loss: {val_metrics['val_loss']:.4f}, L1: {val_metrics['val_l1_loss']:.4f}")
        logging.info(f"LR: {current_lr:.6f}")
        
        # Prepare wandb log dict
        wandb_log = {
            'epoch': epoch,
            'train/loss': train_metrics['loss'],
            'train/l1_loss': train_metrics['l1_loss'],
            'train/kl_loss': train_metrics['kl_loss'],
            'val/loss': val_metrics['val_loss'],
            'val/l1_loss': val_metrics['val_l1_loss'],
            'lr': current_lr,
        }
        
        # Evaluation in environment
        if args.eval_every > 0 and epoch % args.eval_every == 0:
            logging.info(f"\nEvaluating in environment ({args.eval_episodes} episodes)...")
            eval_metrics, video_frames_list = evaluate_in_env(
                model=model,
                device=device,
                action_stats=action_stats,
                num_episodes=args.eval_episodes,
                max_steps=args.eval_max_steps,
                video_dir=eval_video_dir,
                epoch=epoch,
                temporal_ensemble_k=args.temporal_ensemble_k,
            )
            
            logging.info(f"Eval Success Rate: {eval_metrics['eval_success_rate']:.2%}")
            logging.info(f"Eval Avg Reward: {eval_metrics['eval_avg_reward']:.2f}")
            
            wandb_log.update({
                'eval/success_rate': eval_metrics['eval_success_rate'],
                'eval/avg_reward': eval_metrics['eval_avg_reward'],
            })
            
            # Upload video to wandb
            if args.wandb and wandb_run is not None and video_frames_list:
                import wandb
                for ep_idx, frames in enumerate(video_frames_list[:1]):  # Upload first episode only
                    if frames:
                        # (T, H, W, C) -> (T, C, H, W) for wandb
                        video_array = np.stack(frames)
                        if video_array.shape[-1] == 3:
                            video_array = video_array.transpose(0, 3, 1, 2)
                        wandb_log[f'eval/video'] = wandb.Video(
                            video_array, fps=30, format="mp4"
                        )
            
            # Save best model by success rate
            if eval_metrics['eval_success_rate'] > best_success_rate:
                best_success_rate = eval_metrics['eval_success_rate']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['val_loss'],
                    'success_rate': best_success_rate,
                    'config': vars(args),
                }, checkpoint_dir / 'best_success.pt')
                logging.info(f"✓ New best success rate: {best_success_rate:.2%}")
        
        # Wandb logging
        if args.wandb and wandb_run is not None:
            import wandb
            wandb.log(wandb_log)
        
        # Save best model by val loss
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
                'config': vars(args),
            }, checkpoint_dir / 'best_val_loss.pt')
            logging.info(f"✓ New best val loss: {best_val_loss:.4f}")
        
        # Periodic save
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['val_loss'],
                'config': vars(args),
            }, checkpoint_dir / f'checkpoint_epoch_{epoch:04d}.pt')
            logging.info(f"Checkpoint saved: checkpoint_epoch_{epoch:04d}.pt")
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_metrics['val_loss'],
        'config': vars(args),
    }, checkpoint_dir / 'final_model.pt')
    
    logging.info(f"\n{'='*60}")
    logging.info("Training completed!")
    logging.info(f"{'='*60}")
    logging.info(f"Run directory: {run_dir}")
    logging.info(f"Best val loss: {best_val_loss:.4f}")
    logging.info(f"Best success rate: {best_success_rate:.2%}")
    
    if args.wandb and wandb_run is not None:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
