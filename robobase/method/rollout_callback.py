"""
Rollout evaluation callback for BRS policy training.
Periodically evaluates the policy in the actual BigYM environment.
Reuses evaluation logic from eval_brs.py
"""

import numpy as np
import torch
import wandb
import json
import cv2
from pathlib import Path
from typing import Optional, Dict, Any, List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

# Import BigYM environment
import sys
bigym_path = Path(__file__).parent.parent.parent.parent / "bigym"
sys.path.insert(0, str(bigym_path))

from bigym.envs.pick_and_place import SaucepanToHob
from bigym.action_modes import JointPositionActionMode
from bigym.robots.configs.h1 import H1
from bigym.utils.observation_config import ObservationConfig, CameraConfig

# Import video recording
from robobase.video import VideoRecorder


def depth_to_pointcloud(
    depth: np.ndarray,
    rgb: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    depth_scale: float = 1000.0,
    max_depth: float = 3.0
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert depth image to point cloud with RGB colors.
    
    Args:
        depth: Depth image (H, W) in millimeters or raw depth units
        rgb: RGB image (H, W, 3) or (3, H, W) with values [0, 255]
        fx, fy, cx, cy: Camera intrinsic parameters
        depth_scale: Scale factor to convert depth to meters (default 1000 for mm)
        max_depth: Maximum depth in meters to include
    
    Returns:
        xyz: (N, 3) array of 3D points
        rgb_colors: (N, 3) array of RGB colors [0, 1]
    """
    # Handle RGB format
    if rgb.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
        rgb = np.transpose(rgb, (1, 2, 0))
    
    h, w = depth.shape
    
    # Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert depth to meters
    z = depth.astype(np.float32) / depth_scale
    
    # Filter valid depth values
    valid_mask = (z > 0) & (z < max_depth)
    
    # Back-project to 3D
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    # Stack into points
    xyz = np.stack([x, y, z], axis=-1)
    
    # Apply valid mask
    xyz = xyz[valid_mask]
    rgb_valid = rgb[valid_mask]
    
    # Normalize RGB to [0, 1]
    if rgb_valid.dtype == np.uint8:
        rgb_valid = rgb_valid.astype(np.float32) / 255.0
    
    return xyz, rgb_valid


class RolloutEvaluationCallback(Callback):
    """
    Callback for periodic rollout evaluation in BigYM environment.
    
    Evaluates the policy in the actual environment and logs:
    - Success rate
    - Average episode return
    - Episode videos (optional)
    
    Reuses logic from eval_brs.py for consistency.
    """
    
    def __init__(
        self,
        eval_interval: int = 5,
        num_eval_episodes: int = 3,
        log_video: bool = True,
        max_episode_steps: int = 500,
        env_name: str = "SaucepanToHob",
        cameras: List[str] = ["head", "left_wrist", "right_wrist"],
        video_save_dir: Optional[str] = None,
    ):
        """
        Args:
            eval_interval: Evaluate every N epochs
            num_eval_episodes: Number of episodes to run per evaluation
            log_video: Whether to save and log videos
            max_episode_steps: Maximum steps per episode
            env_name: BigYM environment name (default: SaucepanToHob)
            cameras: List of camera names to use for observations
            video_save_dir: Directory to save videos (default: run_dir/eval_videos)
        """
        super().__init__()
        self.eval_interval = eval_interval
        self.num_eval_episodes = num_eval_episodes
        self.log_video = log_video
        self.max_episode_steps = max_episode_steps
        self.env_name = env_name
        self.cameras = cameras
        self.video_save_dir = video_save_dir
        
        self._env = None
        self._last_eval_epoch = -1
        self._video_recorder = None
        self._stats = {}
    
    def setup(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        """Setup callback - create environment and video directory"""
        if stage != "fit":
            return
        
        # Setup video directory
        if self.video_save_dir is None:
            self.video_save_dir = Path(trainer.log_dir) / "eval_videos"
        else:
            self.video_save_dir = Path(self.video_save_dir)
        self.video_save_dir.mkdir(parents=True, exist_ok=True)
        
        # Load normalization stats from config
        self._load_stats(trainer, pl_module)
        
        print(f"\n{'='*80}")
        print(f"Rollout Evaluation Callback Setup")
        print(f"{'='*80}")
        print(f"  Environment: {self.env_name}")
        print(f"  Eval interval: every {self.eval_interval} epochs")
        print(f"  Episodes per eval: {self.num_eval_episodes}")
        print(f"  Max steps per episode: {self.max_episode_steps}")
        print(f"  Log video: {self.log_video}")
        if self.log_video:
            print(f"  Video directory: {self.video_save_dir}")
        print(f"{'='*80}\n")
    
    def _load_stats(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Load normalization statistics"""
        # Try to get config from trainer
        if hasattr(trainer, 'config'):
            config = trainer.config
        else:
            # Load from saved config in run directory
            config_path = Path(trainer.log_dir) / "config.yaml"
            if config_path.exists():
                import yaml
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                print("Warning: No config found, normalization disabled")
                return
        
        # Load stats files
        for stats_type in ['action', 'prop', 'pcd']:
            stats_key = f'{stats_type}_stats_path'
            if stats_key in config:
                stats_path = Path(config[stats_key])
                if stats_path.exists():
                    with open(stats_path, 'r') as f:
                        self._stats[stats_type] = json.load(f)
                    print(f"  ✓ Loaded {stats_type} stats")
    
    def _create_env(self):
        """Create BigYM environment with RGB and depth observations"""
        # Configure cameras with RGB and depth
        camera_names = ['head', 'left_wrist', 'right_wrist']
        cameras = [
            CameraConfig(
                name=cam_name,
                rgb=True,
                depth=True,
                resolution=(480, 640)  # H x W
            )
            for cam_name in camera_names
        ]
        
        obs_config = ObservationConfig(
            cameras=cameras,
            proprioception=True,
            privileged_information=False
        )
        
        if self.env_name == "SaucepanToHob":
            # Always use rgb_array render mode for video recording
            env = SaucepanToHob(
                action_mode=JointPositionActionMode(floating_base=True, absolute=True),
                robot_cls=H1,
                render_mode='rgb_array',  # Required for video recording
                observation_config=obs_config,
            )
        else:
            raise ValueError(f"Unsupported environment: {self.env_name}")
        
        return env
    
    def _obs_to_policy_input(
        self, 
        obs: Dict, 
        config: Dict, 
        device: torch.device
    ) -> Dict:
        """
        Convert BiGym observation to BRS policy input format.
        Adapted from eval_brs.py
        """
        # Extract proprioception from BiGym observation
        prop_data = []
        
        # mobile_base_vel (3): from proprioception_floating_base_actions
        if 'proprioception_floating_base_actions' in obs:
            base_vel = obs['proprioception_floating_base_actions']
            if not isinstance(base_vel, np.ndarray):
                base_vel = np.array(base_vel)
            prop_data.append(base_vel.flatten()[:3])
        
        # torso (1) and arm joints (12): from proprioception
        if 'proprioception' in obs:
            joints = obs['proprioception']
            if not isinstance(joints, np.ndarray):
                joints = np.array(joints)
            prop_data.append(joints.flatten())
        
        # grippers (2): from proprioception_grippers
        if 'proprioception_grippers' in obs:
            grippers = obs['proprioception_grippers']
            if not isinstance(grippers, np.ndarray):
                grippers = np.array(grippers)
            prop_data.append(grippers.flatten())
        
        if len(prop_data) == 0:
            prop = np.zeros(config['prop_dim'], dtype=np.float32)
        else:
            prop = np.concatenate(prop_data, axis=-1).astype(np.float32)
            # Ensure correct size
            if prop.shape[0] != config['prop_dim']:
                if prop.shape[0] < config['prop_dim']:
                    prop = np.pad(prop, (0, config['prop_dim'] - prop.shape[0]))
                else:
                    prop = prop[:config['prop_dim']]
        
        # Extract point cloud from camera observations
        camera_names = ['head', 'left_wrist', 'right_wrist']  # Default cameras
        all_xyz = []
        all_rgb = []
        
        # Camera intrinsics for 640x480 resolution
        # These are typical values and may need adjustment based on actual camera config
        w, h = 640, 480
        fx = fy = 525.0  # Focal length in pixels
        cx = w / 2.0  # Principal point x
        cy = h / 2.0  # Principal point y
        
        for cam_name in camera_names:
            rgb_key = f'rgb_{cam_name}'
            depth_key = f'depth_{cam_name}'
            
            if rgb_key in obs and depth_key in obs:
                rgb_img = obs[rgb_key]  # (3, H, W) or (H, W, 3)
                depth_img = obs[depth_key]  # (H, W)
                
                # Convert depth image to point cloud
                xyz, rgb_colors = depth_to_pointcloud(
                    depth_img, rgb_img, fx, fy, cx, cy,
                    depth_scale=1000.0,  # Assuming depth in mm
                    max_depth=3.0  # 3 meters max depth
                )
                
                all_xyz.append(xyz)
                all_rgb.append(rgb_colors)
        
        # Combine point clouds from all cameras
        if len(all_xyz) > 0:
            pcd_xyz = np.concatenate(all_xyz, axis=0).astype(np.float32)
            pcd_rgb = np.concatenate(all_rgb, axis=0).astype(np.float32)
            
            # Downsample to target number of points
            n_points = config.get('pcd_downsample_points', 6144)
            if pcd_xyz.shape[0] > n_points:
                # Random sampling
                indices = np.random.choice(pcd_xyz.shape[0], n_points, replace=False)
                pcd_xyz = pcd_xyz[indices]
                pcd_rgb = pcd_rgb[indices]
            elif pcd_xyz.shape[0] < n_points:
                # Pad with zeros if not enough points
                pad_size = n_points - pcd_xyz.shape[0]
                pcd_xyz = np.concatenate([pcd_xyz, np.zeros((pad_size, 3), dtype=np.float32)], axis=0)
                pcd_rgb = np.concatenate([pcd_rgb, np.zeros((pad_size, 3), dtype=np.float32)], axis=0)
        else:
            # Fallback to dummy data if no cameras available
            n_points = config.get('pcd_downsample_points', 6144)
            pcd_xyz = np.random.randn(n_points, 3).astype(np.float32) * 0.1
            pcd_rgb = np.random.rand(n_points, 3).astype(np.float32)
        
        # Create policy input
        num_obs = config.get('num_latest_obs', 2)
        policy_input = {
            'odom': {
                'base_velocity': torch.from_numpy(prop[:3][None, None, :]).expand(1, num_obs, -1).to(device),
            },
            'qpos': {
                'torso': torch.from_numpy(prop[3:4][None, None, :]).expand(1, num_obs, -1).to(device),
                'left_arm': torch.from_numpy(prop[4:9][None, None, :]).expand(1, num_obs, -1).to(device),
                'left_gripper': torch.from_numpy(prop[9:10][None, None, :]).expand(1, num_obs, -1).to(device),
                'right_arm': torch.from_numpy(prop[10:15][None, None, :]).expand(1, num_obs, -1).to(device),
                'right_gripper': torch.from_numpy(prop[15:16][None, None, :]).expand(1, num_obs, -1).to(device),
            },
            'pointcloud': {
                'xyz': torch.from_numpy(pcd_xyz[None, None, :, :]).expand(1, num_obs, -1, -1).to(device),
                'rgb': torch.from_numpy(pcd_rgb[None, None, :, :]).expand(1, num_obs, -1, -1).to(device),
            }
        }
        
        return policy_input
    
    def _denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        Denormalize actions from [-1, 1] back to original range.
        Adapted from eval_brs.py
        """
        if 'action' not in self._stats:
            return actions
        
        stats = self._stats['action']
        
        # Extract individual components
        mobile_base_act = actions[:3]
        torso_act = actions[3:4]
        arms_act = actions[4:16]
        
        denormalized_parts = []
        
        # Denormalize mobile_base
        if 'mobile_base' in stats and 'min' in stats['mobile_base']:
            mb_min = np.array(stats['mobile_base']['min'])
            mb_max = np.array(stats['mobile_base']['max'])
            denorm_mb = (mobile_base_act + 1.0) / 2.0 * (mb_max - mb_min + 1e-8) + mb_min
            denormalized_parts.append(denorm_mb)
        else:
            denormalized_parts.append(mobile_base_act)
        
        # Denormalize torso
        if 'torso' in stats and 'min' in stats['torso']:
            t_min = stats['torso']['min']
            t_max = stats['torso']['max']
            denorm_t = (torso_act + 1.0) / 2.0 * (t_max - t_min + 1e-8) + t_min
            denormalized_parts.append(denorm_t)
        else:
            denormalized_parts.append(torso_act)
        
        # Denormalize arms
        if 'arms' in stats and 'min' in stats['arms']:
            a_min = np.array(stats['arms']['min'])
            a_max = np.array(stats['arms']['max'])
            denorm_a = (arms_act + 1.0) / 2.0 * (a_max - a_min + 1e-8) + a_min
            denormalized_parts.append(denorm_a)
        else:
            denormalized_parts.append(arms_act)
        
        return np.concatenate(denormalized_parts, axis=-1)
    
    def _evaluate_policy(
        self, 
        trainer: pl.Trainer, 
        pl_module: pl.LightningModule
    ) -> Dict[str, float]:
        """
        Evaluate policy for multiple episodes.
        Adapted from eval_brs.py
        
        Returns:
            metrics: Dict with success_rate, avg_return, etc.
        """
        if self._env is None:
            self._env = self._create_env()
        
        if self._video_recorder is None:
            self._video_recorder = VideoRecorder(self.video_save_dir if self.log_video else None)
        
        pl_module.eval()
        
        # Get config from trainer
        if hasattr(trainer, 'config'):
            config = trainer.config
        else:
            config_path = Path(trainer.log_dir) / "config.yaml"
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        
        episode_returns = []
        episode_successes = []
        episode_lengths = []
        videos = []
        
        with torch.no_grad():
            for ep_idx in range(self.num_eval_episodes):
                obs, info = self._env.reset()
                
                # Initialize video recorder for this episode
                if self.log_video:
                    self._video_recorder.init(self._env, enabled=True)
                
                episode_reward = 0
                
                for step in range(self.max_episode_steps):
                    # Prepare policy input
                    policy_input = self._obs_to_policy_input(obs, config, pl_module.device)
                    
                    # Get action from policy using inference
                    action_chunks = pl_module.policy.inference(
                        obs=policy_input,
                        return_last_timestep_only=True,
                    )
                    
                    # Extract action (first step of action sequence)
                    # action_chunks: dict with keys ["mobile_base", "torso", "arms"]
                    # Each value has shape [batch, horizon, dim]
                    action_parts = []
                    for key_str in ["mobile_base", "torso", "arms"]:
                        act_tensor = action_chunks[key_str]
                        if act_tensor.dim() == 3:  # [batch, horizon, dim]
                            action_parts.append(act_tensor[0, 0].cpu().numpy())  
                        elif act_tensor.dim() == 2:  # [batch, dim] or [horizon, dim]
                            action_parts.append(act_tensor[0].cpu().numpy())
                        else:  # [dim]
                            action_parts.append(act_tensor.cpu().numpy())
                    
                    action = np.concatenate(action_parts, axis=-1)
                    
                    # Denormalize action
                    if config.get('normalize', False):
                        action = self._denormalize_actions(action)
                    
                    # Clip action to environment action space
                    action = np.clip(action, self._env.action_space.low, self._env.action_space.high)
                    
                    # Execute action
                    obs, reward, terminated, truncated, info = self._env.step(action)
                    episode_reward += reward
                    
                    # Record frame
                    self._video_recorder.record(self._env)
                    
                    if terminated or truncated:
                        break
                
                # Save video
                if self.log_video:
                    video_name = f"epoch_{trainer.current_epoch+1:04d}_ep_{ep_idx+1}.mp4"
                    video_path = self.video_save_dir / video_name
                    self._video_recorder.save(video_name)
                    
                    # Verify video was saved
                    if video_path.exists():
                        file_size = video_path.stat().st_size / (1024 * 1024)  # MB
                        print(f"  ✓ Video saved: {video_path.name} ({len(self._video_recorder.frames)} frames, {file_size:.2f}MB)")
                    else:
                        print(f"  ✗ Video NOT saved: {video_path}")
                
                episode_returns.append(episode_reward)
                episode_lengths.append(step + 1)
                episode_successes.append(1 if info.get('task_success', False) else 0)
                
                print(f"  Episode {ep_idx+1}/{self.num_eval_episodes}: "
                      f"return={episode_reward:.2f}, length={step+1}, "
                      f"success={episode_successes[-1]}")
        
        pl_module.train()
        
        # Compute metrics
        metrics = {
            'eval/avg_return': np.mean(episode_returns),
            'eval/avg_length': np.mean(episode_lengths),
            'eval/success_rate': np.mean(episode_successes),
        }
        
        # Log videos to wandb if available
        if self.log_video:
            self._log_videos_to_wandb(trainer, metrics)
        
        return metrics
    
    def _log_videos_to_wandb(self, trainer: pl.Trainer, metrics: Dict[str, float]):
        """Log videos to wandb"""
        for logger in trainer.loggers:
            if isinstance(logger, pl.loggers.WandbLogger):
                # Find saved videos for current epoch
                video_files = sorted(self.video_save_dir.glob(f"epoch_{trainer.current_epoch+1:04d}_*.mp4"))
                
                if video_files:
                    print(f"  Uploading {len(video_files)} videos to wandb...")
                    wandb_videos = []
                    for video_file in video_files[:3]:  # Log up to 3 videos
                        wandb_videos.append(wandb.Video(str(video_file), fps=20, format="mp4"))
                    
                    logger.experiment.log({
                        "eval/videos": wandb_videos,
                        "epoch": trainer.current_epoch + 1,
                    })
    
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Called at the end of each training epoch"""
        # Only evaluate at specified intervals
        if (trainer.current_epoch + 1) % self.eval_interval != 0:
            return
        
        # Avoid duplicate evaluation
        if trainer.current_epoch == self._last_eval_epoch:
            return
        
        self._last_eval_epoch = trainer.current_epoch
        
        print(f"\n{'='*80}")
        print(f"Rollout Evaluation (Epoch {trainer.current_epoch + 1})")
        print(f"{'='*80}")
        
        # Run evaluation
        try:
            metrics = self._evaluate_policy(trainer, pl_module)
            
            # Log metrics to wandb/logger
            for key, value in metrics.items():
                pl_module.log(
                    key,
                    value,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
            
            print(f"\nEvaluation Results:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")
            print(f"{'='*80}\n")
        except Exception as e:
            print(f"Error during rollout evaluation: {e}")
            import traceback
            traceback.print_exc()
    
    def teardown(self, trainer: pl.Trainer, pl_module: pl.LightningModule, stage: str):
        """Cleanup - close environment"""
        if self._env is not None:
            self._env.close()
            self._env = None

