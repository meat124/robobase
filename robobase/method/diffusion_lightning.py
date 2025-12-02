"""
Diffusion Policy Training with PyTorch Lightning
RGB-based training for BigYM demonstrations, using same data format as ACT.

Uses ACTHdf5Dataset for data loading (RGB + 16D proprioception + 16D actions).
"""

import os
import sys
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, RichModelSummary
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from torch.utils.data import DataLoader
import copy
import json
import logging

from diffusers import DDIMScheduler

# Import ACT dataset (same as ACT training)
from robobase.method.act_hdf5_dataset import ACTHdf5Dataset, ACT_PROP_DIM, ACT_ACTION_DIM


# ============================================================================
# Diffusion Policy Components
# ============================================================================

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Conv1dBlock(nn.Module):
    """Conv1d --> GroupNorm --> Mish"""
    
    def __init__(self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(inp_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Downsample1d(nn.Module):
    """Downsampling layer for 1D sequences."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    """Upsampling layer for 1D sequences."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ConditionalResidualBlock1D(nn.Module):
    """Residual block with FiLM conditioning."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        
        self.blocks = nn.ModuleList([
            Conv1dBlock(in_channels, out_channels, kernel_size, n_groups),
            Conv1dBlock(out_channels, out_channels, kernel_size, n_groups),
        ])
        
        # FiLM conditioning
        cond_channels = out_channels * 2 if cond_predict_scale else out_channels
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
        )
        
        # Residual connection
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )
    
    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) features
            cond: (B, cond_dim) conditioning
        Returns:
            (B, out_channels, T)
        """
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond).unsqueeze(-1)  # (B, cond_channels, 1)
        
        if self.cond_predict_scale:
            scale, bias = embed.chunk(2, dim=1)
            out = scale * out + bias
        else:
            out = out + embed
        
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUNet1D(nn.Module):
    """1D U-Net for diffusion noise prediction with proper skip connections.
    
    This is a standard U-Net architecture with:
    - Encoder path with downsampling
    - Skip connections between encoder and decoder
    - Decoder path with upsampling
    - FiLM conditioning at each residual block
    """
    
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        diffusion_step_embed_dim: int = 256,
        down_dims: List[int] = [256, 512, 1024],
        kernel_size: int = 5,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        
        # Diffusion timestep embedding
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            SinusoidalPosEmb(dsed),
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),
        )
        
        # Conditioning dimension = diffusion step emb + global condition
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim
        
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        
        # Middle (bottleneck) blocks
        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList([
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale
            ),
            ConditionalResidualBlock1D(
                mid_dim, mid_dim, cond_dim, kernel_size, n_groups, cond_predict_scale
            ),
        ])
        
        # Encoder (down) modules
        self.down_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            self.down_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(
                        dim_in, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale
                    ),
                    ConditionalResidualBlock1D(
                        dim_out, dim_out, cond_dim, kernel_size, n_groups, cond_predict_scale
                    ),
                    Downsample1d(dim_out) if not is_last else nn.Identity(),
                ])
            )
        
        # Decoder (up) modules
        self.up_modules = nn.ModuleList()
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            self.up_modules.append(
                nn.ModuleList([
                    ConditionalResidualBlock1D(
                        dim_out * 2, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale
                    ),
                    ConditionalResidualBlock1D(
                        dim_in, dim_in, cond_dim, kernel_size, n_groups, cond_predict_scale
                    ),
                    Upsample1d(dim_in) if not is_last else nn.Identity(),
                ])
            )
        
        # Final convolution
        self.final_conv = nn.Sequential(
            Conv1dBlock(start_dim, start_dim, kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        timestep: torch.Tensor,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, action_dim) noisy actions
            timestep: (B,) diffusion timesteps
            global_cond: (B, global_cond_dim) conditioning features
        Returns:
            (B, T, action_dim) predicted noise
        """
        # Reshape for conv1d: (B, T, C) -> (B, C, T)
        x = x.permute(0, 2, 1)
        
        # Timestep embedding
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=x.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(x.device)
        timesteps = timesteps.expand(x.shape[0])
        
        global_feature = self.diffusion_step_encoder(timesteps)  # (B, dsed)
        
        # Combine with global condition
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)
        
        # Encoder path with skip connections
        h = []
        for idx, (resnet, resnet2, downsample) in enumerate(self.down_modules):
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)  # Save for skip connection
            x = downsample(x)
        
        # Middle (bottleneck)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        
        # Decoder path with skip connections
        for idx, (resnet, resnet2, upsample) in enumerate(self.up_modules):
            x = torch.cat((x, h.pop()), dim=1)  # Concatenate skip connection
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Reshape back: (B, C, T) -> (B, T, C)
        x = x.permute(0, 2, 1)
        return x


class ResNetEncoder(nn.Module):
    """ResNet18-based encoder for RGB images (same as ACT)."""
    
    VISUAL_OBS_MEAN = [0.485, 0.456, 0.406]
    VISUAL_OBS_STD = [0.229, 0.224, 0.225]
    
    def __init__(self, output_dim: int = 256, pretrained: bool = True):
        super().__init__()
        import torchvision.models as models
        
        # Use weights parameter for newer torchvision
        try:
            from torchvision.models import ResNet18_Weights
            resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        except ImportError:
            resnet = models.resnet18(pretrained=pretrained)
        
        # Remove last FC layer
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Projection layer
        self.proj = nn.Linear(512, output_dim)
        
        # Image normalization buffers
        self.register_buffer(
            'img_mean',
            torch.tensor(self.VISUAL_OBS_MEAN).view(1, 3, 1, 1)
        )
        self.register_buffer(
            'img_std',
            torch.tensor(self.VISUAL_OBS_STD).view(1, 3, 1, 1)
        )
    
    def normalize_images(self, images: torch.Tensor) -> torch.Tensor:
        """Normalize images to ImageNet statistics."""
        # images: (B, C, H, W) in [0, 255] uint8 or float
        if images.dtype == torch.uint8:
            images = images.float()
        images = images / 255.0
        images = (images - self.img_mean) / self.img_std
        return images
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) RGB image in [0, 255]
        Returns:
            (B, output_dim) features
        """
        x = self.normalize_images(x)
        h = self.backbone(x)  # (B, 512, 1, 1)
        h = h.view(h.size(0), -1)  # (B, 512)
        out = self.proj(h)  # (B, output_dim)
        return out


# ============================================================================
# Diffusion Policy (RGB-based, same data as ACT)
# ============================================================================

class DiffusionPolicyRGB(nn.Module):
    """Diffusion Policy with RGB image encoder (same data format as ACT)."""
    
    def __init__(
        self,
        prop_dim: int = 16,
        action_dim: int = 16,
        action_horizon: int = 16,
        frame_stack: int = 1,
        encoder_output_dim: int = 256,
        diffusion_step_embed_dim: int = 256,
        unet_down_dims: List[int] = [256, 512, 1024],
        unet_kernel_size: int = 5,
        unet_n_groups: int = 8,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 16,
        prop_hidden_dim: int = 256,
    ):
        super().__init__()
        
        self.prop_dim = prop_dim
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.frame_stack = frame_stack
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        
        # Proprioception encoder
        self.prop_encoder = nn.Sequential(
            nn.Linear(prop_dim * frame_stack, prop_hidden_dim),
            nn.ReLU(),
            nn.Linear(prop_hidden_dim, prop_hidden_dim),
            nn.ReLU(),
            nn.Linear(prop_hidden_dim, encoder_output_dim),
        )
        
        # RGB encoder (ResNet18)
        self.rgb_encoder = ResNetEncoder(output_dim=encoder_output_dim, pretrained=True)
        
        # Conditioning dimension: prop + rgb
        cond_dim = encoder_output_dim * 2
        
        # U-Net for noise prediction
        self.unet = ConditionalUNet1D(
            input_dim=action_dim,
            global_cond_dim=cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=unet_down_dims,
            kernel_size=unet_kernel_size,
            n_groups=unet_n_groups,
        )
        
        # Noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
    
    def encode_observation(
        self,
        prop: torch.Tensor,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        """Encode observations to conditioning features.
        
        Args:
            prop: (B, frame_stack * prop_dim) proprioception (already flattened)
            rgb: (B, frame_stack, C, H, W) or (B, C, H, W) RGB images
        
        Returns:
            (B, cond_dim) conditioning features
        """
        B = prop.shape[0]
        
        # Prop is already flattened from ACT dataset
        prop_flat = prop  # (B, frame_stack * prop_dim)
        
        if rgb.dim() == 5:
            # Use last frame for RGB
            rgb = rgb[:, -1]  # (B, C, H, W)
        
        # Encode
        prop_feat = self.prop_encoder(prop_flat)  # (B, encoder_output_dim)
        rgb_feat = self.rgb_encoder(rgb)  # (B, encoder_output_dim)
        
        # Concatenate
        cond = torch.cat([prop_feat, rgb_feat], dim=-1)  # (B, 2 * encoder_output_dim)
        
        return cond
    
    def forward(
        self,
        prop: torch.Tensor,
        rgb: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass.
        
        Args:
            prop: (B, frame_stack, prop_dim) or (B, prop_dim)
            rgb: (B, frame_stack, C, H, W) or (B, C, H, W)
            action: (B, action_horizon, action_dim) target actions
        
        Returns:
            (noise_pred, noise) tuple for loss computation
        """
        B = prop.shape[0]
        device = prop.device
        
        # Encode observations
        cond = self.encode_observation(prop, rgb)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.num_train_timesteps, (B,), device=device
        ).long()
        
        # Sample noise
        noise = torch.randn_like(action)
        
        # Add noise to actions
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)
        
        # Predict noise
        noise_pred = self.unet(noisy_action, timesteps, cond)
        
        return noise_pred, noise
    
    @torch.no_grad()
    def infer(
        self,
        prop: torch.Tensor,
        rgb: torch.Tensor,
    ) -> torch.Tensor:
        """Inference (denoising) to generate actions.
        
        Args:
            prop: (B, frame_stack, prop_dim) or (B, prop_dim)
            rgb: (B, frame_stack, C, H, W) or (B, C, H, W)
        
        Returns:
            (B, action_horizon, action_dim) predicted actions
        """
        B = prop.shape[0]
        device = prop.device
        
        # Encode observations
        cond = self.encode_observation(prop, rgb)
        
        # Initialize from Gaussian noise
        noisy_action = torch.randn(
            (B, self.action_horizon, self.action_dim), device=device
        )
        
        # Denoise
        self.noise_scheduler.set_timesteps(self.num_inference_steps)
        
        for t in self.noise_scheduler.timesteps:
            timestep = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.unet(noisy_action, timestep, cond)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_action,
            ).prev_sample
        
        return noisy_action


# ============================================================================
# Lightning Module
# ============================================================================

class DiffusionLightningModule(pl.LightningModule):
    """PyTorch Lightning module for Diffusion Policy training (RGB-based)."""
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Create policy
        self.policy = DiffusionPolicyRGB(
            prop_dim=config.get('prop_dim', 16),
            action_dim=config.get('action_dim', 16),
            action_horizon=config.get('action_sequence', 16),
            frame_stack=config.get('frame_stack', 1),
            encoder_output_dim=config.get('encoder_output_dim', 256),
            diffusion_step_embed_dim=config.get('diffusion_step_embed_dim', 256),
            unet_down_dims=config.get('unet_down_dims', [256, 512, 1024]),
            unet_kernel_size=config.get('unet_kernel_size', 5),
            unet_n_groups=config.get('unet_n_groups', 8),
            num_train_timesteps=config.get('num_train_timesteps', 100),
            num_inference_steps=config.get('num_inference_steps', 16),
            prop_hidden_dim=config.get('prop_hidden_dim', 256),
        )
        
        # EMA for stable inference (use simple custom EMA)
        self.ema_decay = config.get('ema_decay', 0.9999)
        self.ema_params = {name: p.clone().detach() for name, p in self.policy.named_parameters()}
        self.ema_policy = None  # Created during inference
        
        # Loss
        self.loss_fn = nn.MSELoss()
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass for training."""
        # ACT dataset batch format:
        # - 'rgb_head': (B, FS, C, H, W) float32
        # - 'low_dim_state': (B, FS * prop_dim) float32 (flattened)
        # - 'action': (B, action_seq, action_dim) float32
        
        rgb = batch['rgb_head']  # (B, FS, C, H, W)
        prop = batch['low_dim_state']  # (B, FS * prop_dim) - already flattened
        action = batch['action']  # (B, action_seq, action_dim)
        
        # Forward pass
        noise_pred, noise = self.policy(prop, rgb, action)
        
        return noise_pred, noise
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        noise_pred, noise = self(batch)
        
        loss = self.loss_fn(noise_pred, noise)
        
        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            logging.warning(f"NaN/Inf loss at batch {batch_idx}")
            loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
        
        # Update EMA (manual update)
        with torch.no_grad():
            for name, param in self.policy.named_parameters():
                if name in self.ema_params:
                    # Move EMA params to same device as model params if needed
                    if self.ema_params[name].device != param.device:
                        self.ema_params[name] = self.ema_params[name].to(param.device)
                    self.ema_params[name].mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)
        
        # Logging
        self.log('train/loss', loss, prog_bar=True, on_step=True, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        noise_pred, noise = self(batch)
        
        loss = self.loss_fn(noise_pred, noise)
        
        self.log('val/loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        lr = self.config.get('lr', 1e-4)
        weight_decay = self.config.get('weight_decay', 0.01)
        
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Cosine scheduler with warmup
        if self.config.get('use_cosine_lr', True):
            warmup_steps = self.config.get('lr_warmup_steps', 500)
            total_steps = self.config.get('total_steps', 100000)
            min_lr = self.config.get('lr_min', 1e-6)
            
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
            
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=min_lr,
            )
            scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps],
            )
            
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                },
            }
        
        return optimizer


# ============================================================================
# RGB Rollout Evaluation
# ============================================================================

# Environment runs at 500Hz, but policy outputs actions at 50Hz (decimation=10)
ENV_DECIMATION = 10


def create_eval_env():
    """Create BigYM environment for evaluation."""
    try:
        # Import BigYM
        bigym_path = Path(__file__).parent.parent.parent.parent / "bigym"
        sys.path.insert(0, str(bigym_path))
        
        from bigym.envs.pick_and_place import SaucepanToHob
        from bigym.action_modes import JointPositionActionMode
        from bigym.robots.configs.h1 import H1, PelvisDof
        from bigym.utils.observation_config import ObservationConfig, CameraConfig
        
        # Configure cameras with RGB
        cameras = [
            CameraConfig(
                name='head',
                rgb=True,
                depth=False,
                resolution=(480, 640)
            )
        ]
        
        obs_config = ObservationConfig(
            cameras=cameras,
            proprioception=True,
            privileged_information=False
        )
        
        floating_dofs = [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
        
        env = SaucepanToHob(
            action_mode=JointPositionActionMode(
                floating_base=True,
                absolute=True,
                floating_dofs=floating_dofs
            ),
            robot_cls=H1,
            render_mode='rgb_array',
            observation_config=obs_config,
        )
        
        return env
    except Exception as e:
        logging.warning(f"Failed to create eval environment: {e}")
        return None


def save_video(frames: List[np.ndarray], path: Path, fps: int = 30) -> bool:
    """Save frames as MP4 video."""
    if not frames:
        return False
    
    try:
        import imageio
        writer = imageio.get_writer(
            str(path),
            fps=fps,
            codec='libx264',
            quality=8,
            pixelformat='yuv420p',
            output_params=['-preset', 'fast'],
        )
        
        for frame in frames:
            if len(frame.shape) == 2:
                frame = np.stack([frame] * 3, axis=-1)
            writer.append_data(frame)
        
        writer.close()
        return True
    except Exception as e:
        logging.warning(f"imageio failed: {e}, trying cv2...")
    
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


def unnormalize_action(action_norm: np.ndarray, action_stats: Dict) -> np.ndarray:
    """Unnormalize action from [-1, 1] to original range."""
    action_min = action_stats['min']
    action_max = action_stats['max']
    action = (action_norm + 1.0) / 2.0 * (action_max - action_min) + action_min
    return action


@torch.no_grad()
def evaluate_in_env(
    policy: DiffusionPolicyRGB,
    device: torch.device,
    action_stats: Dict,
    num_episodes: int = 3,
    max_steps: int = 500,
    video_dir: Optional[Path] = None,
    epoch: int = 0,
    decimation: int = ENV_DECIMATION,
) -> Tuple[Dict, List]:
    """
    Evaluate Diffusion Policy in BigYM environment.
    
    Args:
        policy: DiffusionPolicyRGB model
        device: torch device
        action_stats: Action statistics for unnormalization
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode (at 50Hz)
        video_dir: Directory to save videos
        epoch: Current epoch for video naming
        decimation: Env steps per policy action
    
    Returns:
        metrics: Dict with success_rate, avg_reward
        video_frames_list: List of frame lists
    """
    env = create_eval_env()
    if env is None:
        return {'eval_success_rate': 0.0, 'eval_avg_reward': 0.0}, []
    
    policy.eval()
    
    successes = []
    total_rewards = []
    video_frames_list = []
    
    # QPOS indices for arms
    QPOS_LEFT_ARM = [0, 1, 2, 3, 12]
    QPOS_RIGHT_ARM = [13, 14, 15, 16, 25]
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        
        frames = []
        episode_reward = 0.0
        done = False
        
        # Action queue for action chunking
        action_queue = []
        
        for step in range(max_steps):
            # Render frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            
            # Prepare RGB observation
            rgb = obs['rgb_head']
            if len(rgb.shape) == 3 and rgb.shape[-1] == 3:
                rgb = rgb.transpose(2, 0, 1)  # (H, W, C) -> (C, H, W)
            
            # Resize to 224x224
            if rgb.shape[1] != 224 or rgb.shape[2] != 224:
                import cv2
                rgb_hwc = rgb.transpose(1, 2, 0)
                rgb_resized = cv2.resize(rgb_hwc, (224, 224))
                rgb = rgb_resized.transpose(2, 0, 1)
            
            # Get proprioception (16D)
            qpos = obs['proprioception'][:30]
            floating_base = obs['proprioception_floating_base']
            grippers = obs['proprioception_grippers']
            
            left_arm = qpos[QPOS_LEFT_ARM]
            right_arm = qpos[QPOS_RIGHT_ARM]
            
            prop_16d = np.concatenate([floating_base, left_arm, right_arm, grippers])
            
            # Get new action chunk if queue is empty
            if len(action_queue) == 0:
                rgb_tensor = torch.from_numpy(rgb).unsqueeze(0).float().to(device)
                prop_tensor = torch.from_numpy(prop_16d).unsqueeze(0).float().to(device)
                
                # Infer action sequence
                action_pred = policy.infer(prop_tensor, rgb_tensor)  # (1, T, 16)
                action_chunk = action_pred[0].cpu().numpy()  # (T, 16)
                
                # Use first 4 actions (temporal ensemble)
                action_queue = list(action_chunk[:4])
            
            # Get next action
            action_16d_norm = action_queue.pop(0)
            
            # Unnormalize action
            action_16d = unnormalize_action(action_16d_norm, action_stats)
            
            # Apply decimation for 500Hz/50Hz mismatch
            action_500hz = action_16d.copy()
            action_500hz[0:4] = action_16d[0:4] / decimation
            
            # Execute decimation env steps
            step_reward = 0.0
            for _ in range(decimation):
                action = np.clip(action_500hz, env.action_space.low, env.action_space.high)
                obs, reward, terminated, truncated, info = env.step(action)
                step_reward += reward
                done = terminated or truncated
                
                if done:
                    break
            
            episode_reward += step_reward
            
            if done:
                break
        
        success = info.get('success', False) or info.get('is_success', False)
        successes.append(float(success))
        total_rewards.append(episode_reward)
        
        if frames:
            video_frames_list.append(frames)
        
        logging.info(f"  Episode {ep+1}: steps={step+1}, reward={episode_reward:.2f}, success={success}")
    
    env.close()
    
    # Save videos
    if video_dir is not None and video_frames_list:
        video_dir = Path(video_dir)
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


class RGBRolloutEvaluationCallback(pl.callbacks.Callback):
    """
    Callback for periodic rollout evaluation in BigYM environment (RGB-based).
    """
    
    def __init__(
        self,
        eval_interval: int = 50,
        num_eval_episodes: int = 3,
        max_episode_steps: int = 500,
        log_video: bool = True,
        video_save_dir: Optional[str] = None,
        action_stats_path: Optional[str] = None,
    ):
        super().__init__()
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.max_episode_steps = max_episode_steps
        self.log_video = log_video
        self.video_save_dir = video_save_dir
        self.action_stats_path = action_stats_path
        self._action_stats = None
        self._best_success_rate = 0.0
    
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        if stage != "fit":
            return
        
        # Setup video directory
        if self.video_save_dir is None:
            self.video_save_dir = Path(trainer.log_dir) / "eval_videos"
        else:
            self.video_save_dir = Path(self.video_save_dir)
        self.video_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load action stats
        if self.action_stats_path:
            with open(self.action_stats_path, 'r') as f:
                raw_stats = json.load(f)
            
            if 'min' in raw_stats and isinstance(raw_stats['min'], list):
                self._action_stats = {
                    'min': np.array(raw_stats['min'], dtype=np.float32),
                    'max': np.array(raw_stats['max'], dtype=np.float32),
                }
            elif 'full' in raw_stats:
                self._action_stats = {
                    'min': np.array(raw_stats['full']['min'], dtype=np.float32),
                    'max': np.array(raw_stats['full']['max'], dtype=np.float32),
                }
            else:
                logging.warning("Unknown action stats format, eval disabled")
                self._action_stats = None
        
        logging.info(f"\n{'='*60}")
        logging.info("RGB Rollout Evaluation Callback Setup")
        logging.info(f"  Eval interval: every {self.eval_interval} epochs")
        logging.info(f"  Episodes per eval: {self.num_eval_episodes}")
        logging.info(f"  Max steps per episode: {self.max_episode_steps}")
        logging.info(f"  Log video: {self.log_video}")
        logging.info(f"  Video directory: {self.video_save_dir}")
        logging.info(f"{'='*60}\n")
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        epoch = trainer.current_epoch
        
        if epoch % self.eval_interval != 0 or epoch == 0:
            return
        
        if self._action_stats is None:
            logging.warning("Action stats not loaded, skipping eval")
            return
        
        logging.info(f"\n[Epoch {epoch}] Running rollout evaluation...")
        
        # Get policy from Lightning module
        policy = pl_module.policy
        device = pl_module.device
        
        metrics, video_frames_list = evaluate_in_env(
            policy=policy,
            device=device,
            action_stats=self._action_stats,
            num_episodes=self.num_eval_episodes,
            max_steps=self.max_episode_steps,
            video_dir=self.video_save_dir if self.log_video else None,
            epoch=epoch,
        )
        
        logging.info(f"Eval Success Rate: {metrics['eval_success_rate']:.2%}")
        logging.info(f"Eval Avg Reward: {metrics['eval_avg_reward']:.2f}")
        
        # Log to trainer
        pl_module.log('eval/success_rate', metrics['eval_success_rate'])
        pl_module.log('eval/avg_reward', metrics['eval_avg_reward'])
        
        # Log video to wandb
        if self.log_video and video_frames_list:
            for logger in trainer.loggers:
                if isinstance(logger, WandbLogger):
                    import wandb
                    for ep_idx, frames in enumerate(video_frames_list[:1]):
                        if frames:
                            video_array = np.stack(frames)
                            if video_array.shape[-1] == 3:
                                video_array = video_array.transpose(0, 3, 1, 2)
                            logger.experiment.log({
                                'eval/video': wandb.Video(video_array, fps=30, format="mp4"),
                                'epoch': epoch,
                            })
        
        # Track best success rate
        if metrics['eval_success_rate'] > self._best_success_rate:
            self._best_success_rate = metrics['eval_success_rate']
            logging.info(f"✓ New best success rate: {self._best_success_rate:.2%}")
            
            # Save best model
            checkpoint_dir = Path(trainer.log_dir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'state_dict': pl_module.state_dict(),
                'success_rate': self._best_success_rate,
            }, checkpoint_dir / 'best_success.pt')


# ============================================================================
# Data Module
# ============================================================================

class DiffusionDataModule(pl.LightningDataModule):
    """DataModule for Diffusion Policy using ACT dataset."""
    
    def __init__(
        self,
        hdf5_path: str,
        action_stats_path: Optional[str] = None,
        image_size: Tuple[int, int] = (224, 224),
        frame_stack: int = 1,
        action_sequence: int = 16,
        batch_size: int = 64,
        val_batch_size: int = 64,
        val_split_ratio: float = 0.1,
        num_workers: int = 4,
        normalize_actions: bool = True,
        camera: str = 'head',
        preload_all: bool = False,
    ):
        super().__init__()
        self.hdf5_path = hdf5_path
        self.action_stats_path = action_stats_path
        self.image_size = image_size
        self.frame_stack = frame_stack
        self.action_sequence = action_sequence
        self.batch_size = batch_size
        self.val_batch_size = val_batch_size
        self.val_split_ratio = val_split_ratio
        self.num_workers = num_workers
        self.normalize_actions = normalize_actions
        self.camera = camera
        self.preload_all = preload_all
    
    def setup(self, stage: Optional[str] = None):
        import h5py
        
        # Get all demo IDs
        with h5py.File(self.hdf5_path, 'r') as f:
            all_demo_ids = sorted([
                int(k.split('_')[1]) 
                for k in f.keys() 
                if k.startswith('demo_')
            ])
        
        # Train/val split
        n_val = max(1, int(len(all_demo_ids) * self.val_split_ratio))
        np.random.seed(42)
        np.random.shuffle(all_demo_ids)
        
        val_demo_ids = all_demo_ids[:n_val]
        train_demo_ids = all_demo_ids[n_val:]
        
        logging.info(f"Train demos: {len(train_demo_ids)}, Val demos: {len(val_demo_ids)}")
        
        # Create datasets
        self.train_dataset = ACTHdf5Dataset(
            hdf5_path=self.hdf5_path,
            action_stats_path=self.action_stats_path,
            demo_ids=train_demo_ids,
            image_size=self.image_size,
            frame_stack=self.frame_stack,
            action_sequence=self.action_sequence,
            normalize_actions=self.normalize_actions,
            camera=self.camera,
            preload_all=self.preload_all,
        )
        
        self.val_dataset = ACTHdf5Dataset(
            hdf5_path=self.hdf5_path,
            action_stats_path=self.action_stats_path,
            demo_ids=val_demo_ids,
            image_size=self.image_size,
            frame_stack=self.frame_stack,
            action_sequence=self.action_sequence,
            normalize_actions=self.normalize_actions,
            camera=self.camera,
            preload_all=self.preload_all,
        )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers if not self.preload_all else 0,
            pin_memory=True,
            drop_last=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers if not self.preload_all else 0,
            pin_memory=True,
        )


# ============================================================================
# Main Training Function
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Diffusion Policy Training (RGB)")
    parser.add_argument('--hdf5-path', type=str, required=True, help='Path to HDF5 demos file')
    parser.add_argument('--action-stats-path', type=str, help='Path to action_stats.json')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='Max epochs')
    parser.add_argument('--action-sequence', type=int, default=16, help='Action prediction horizon')
    parser.add_argument('--image-size', type=int, default=224, help='Image size')
    parser.add_argument('--frame-stack', type=int, default=1, help='Frame stack')
    parser.add_argument('--num-workers', type=int, default=4, help='DataLoader workers')
    parser.add_argument('--preload-all', action='store_true', help='Preload all data into RAM')
    parser.add_argument('--run-name', type=str, default='diffusion_run', help='Run name')
    parser.add_argument('--wandb', action='store_true', help='Enable W&B logging')
    parser.add_argument('--wandb-project', type=str, default='diffusion_policy', help='W&B project')
    parser.add_argument('--eval-every', type=int, default=50, help='Eval every N epochs (0 to disable)')
    parser.add_argument('--eval-episodes', type=int, default=3, help='Number of eval episodes')
    parser.add_argument('--eval-max-steps', type=int, default=500, help='Max steps per eval episode')
    parser.add_argument('--log-eval-video', action='store_true', help='Log eval videos to wandb')
    parser.add_argument('--camera', type=str, default='head', help='Camera to use')
    parser.add_argument('--num-train-timesteps', type=int, default=100, help='Diffusion timesteps')
    parser.add_argument('--num-inference-steps', type=int, default=16, help='Inference steps')
    parser.add_argument('--gradient-clip', type=float, default=1.0, help='Gradient clip value')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    
    # Config
    config = {
        'hdf5_path': args.hdf5_path,
        'action_stats_path': args.action_stats_path,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'epochs': args.epochs,
        'action_sequence': args.action_sequence,
        'image_size': (args.image_size, args.image_size),
        'frame_stack': args.frame_stack,
        'num_workers': args.num_workers,
        'preload_all': args.preload_all,
        'run_name': args.run_name,
        'camera': args.camera,
        'prop_dim': ACT_PROP_DIM,  # 16
        'action_dim': ACT_ACTION_DIM,  # 16
        'encoder_output_dim': 256,
        'diffusion_step_embed_dim': 256,
        'unet_down_dims': [256, 512, 1024],
        'unet_kernel_size': 5,
        'unet_n_groups': 8,
        'num_train_timesteps': args.num_train_timesteps,
        'num_inference_steps': args.num_inference_steps,
        'use_cosine_lr': True,
        'lr_warmup_steps': 500,
        'weight_decay': 0.01,
    }
    
    print("=" * 60)
    print("Diffusion Policy Training (RGB)")
    print("=" * 60)
    print(f"HDF5: {args.hdf5_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Action sequence: {args.action_sequence}")
    print(f"Camera: {args.camera}")
    print(f"Diffusion timesteps: {args.num_train_timesteps}")
    print("=" * 60)
    
    # Create data module
    datamodule = DiffusionDataModule(
        hdf5_path=args.hdf5_path,
        action_stats_path=args.action_stats_path,
        image_size=(args.image_size, args.image_size),
        frame_stack=args.frame_stack,
        action_sequence=args.action_sequence,
        batch_size=args.batch_size,
        val_batch_size=args.batch_size,
        val_split_ratio=0.1,
        num_workers=args.num_workers,
        normalize_actions=True,
        camera=args.camera,
        preload_all=args.preload_all,
    )
    
    # Create model
    model = DiffusionLightningModule(config)
    
    # Callbacks
    checkpoint_dir = Path(f"runs/{args.run_name}/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename='epoch_{epoch:04d}-val_loss_{val/loss:.4f}',
            save_top_k=3,
            monitor='val/loss',
            mode='min',
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]
    
    # Add rollout evaluation callback if enabled
    if args.eval_every > 0:
        eval_callback = RGBRolloutEvaluationCallback(
            eval_interval=args.eval_every,
            num_eval_episodes=args.eval_episodes,
            max_episode_steps=args.eval_max_steps,
            log_video=args.log_eval_video,
            video_save_dir=f'runs/{args.run_name}/eval_videos',
            action_stats_path=args.action_stats_path,
        )
        callbacks.append(eval_callback)
        print(f"✓ Rollout evaluation enabled:")
        print(f"  - Interval: every {args.eval_every} epochs")
        print(f"  - Episodes: {args.eval_episodes}")
        print(f"  - Log video: {args.log_eval_video}")
    
    # Loggers
    loggers = [
        TensorBoardLogger(save_dir=f'runs/{args.run_name}', name='tb_logs'),
        CSVLogger(save_dir=f'runs/{args.run_name}', name='csv_logs'),
    ]
    
    if args.wandb:
        loggers.append(WandbLogger(
            project=args.wandb_project,
            name=args.run_name,
            save_dir=f'runs/{args.run_name}',
        ))
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices=1,
        precision='32',
        gradient_clip_val=args.gradient_clip,
        callbacks=callbacks,
        logger=loggers,
        log_every_n_steps=10,
        val_check_interval=1.0,
        check_val_every_n_epoch=max(1, args.eval_every // 5) if args.eval_every > 0 else 50,
        enable_progress_bar=True,
    )
    
    # Train
    trainer.fit(model, datamodule)
    
    print("=" * 60)
    print("Training completed!")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
