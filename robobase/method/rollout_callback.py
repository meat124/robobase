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
from bigym.envs.manipulation import FlipCup
from bigym.action_modes import JointPositionActionMode
from bigym.robots.configs.h1 import H1, PelvisDof
from bigym.utils.observation_config import ObservationConfig, CameraConfig

# Import video recording
from robobase.video import VideoRecorder

# Environment runs at 500Hz, but policy outputs actions at 50Hz (decimation=10)
# Floating base actions in demo are SUM of 10 x 500Hz deltas
# To replay correctly: divide floating base by 10, execute 10 env steps per policy action
ENV_DECIMATION = 10


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
        
        # Store config for later use
        self._config = config
        
        # Load stats files
        stats_paths = {
            'action': config.get('action_stats_path'),
            'prop': config.get('prop_stats_path'),
            'pcd': config.get('pcd_stats_path'),
        }
        
        for stats_type, stats_path in stats_paths.items():
            if stats_path:
                stats_path = Path(stats_path)
                if stats_path.exists():
                    with open(stats_path, 'r') as f:
                        self._stats[stats_type] = json.load(f)
                    print(f"  ✓ Loaded {stats_type} stats from {stats_path}")
                else:
                    print(f"  ⚠ {stats_type} stats not found: {stats_path}")
    
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
        
        # Include PelvisDof.Z (torso) for height control - required for manipulation tasks
        # BRS Policy uses 16D actions: [X, Y, Z, RZ] + arms(10) + grippers(2)
        floating_dofs = [PelvisDof.X, PelvisDof.Y, PelvisDof.Z, PelvisDof.RZ]
        
        # Select environment class based on env_name
        if self.env_name == "SaucepanToHob":
            env_cls = SaucepanToHob
        elif self.env_name == "FlipCup":
            env_cls = FlipCup
        else:
            raise ValueError(f"Unsupported environment: {self.env_name}")
        
        # Always use rgb_array render mode for video recording
        env = env_cls(
            action_mode=JointPositionActionMode(
                floating_base=True, 
                absolute=True,
                floating_dofs=floating_dofs
            ),
            robot_cls=H1,
            render_mode='rgb_array',  # Required for video recording
            observation_config=obs_config,
        )
        
        return env
    
    def _brs_to_bigym_action(self, brs_action: np.ndarray) -> np.ndarray:
        """
        Convert BRS 16D action to BigYM 16D action format.
        
        BRS 16D structure (from policy output, concatenated):
            [0:3]   mobile_base (3D): dx, dy, drz
            [3:4]   torso (1D): dz
            [4:16]  arms (12D): left_arm(5) + right_arm(5) + grippers(2)
        
        BigYM 16D structure (with floating_dofs=[X, Y, Z, RZ]):
            [0:4]   floating_base (4D): dx, dy, dz, drz
            [4:9]   left_arm (5D): joint positions
            [9:14]  right_arm (5D): joint positions
            [14:16] grippers (2D): [left_gripper, right_gripper]
        """
        # Extract BRS action components
        mobile_base = brs_action[0:3]     # dx, dy, drz
        torso = brs_action[3:4]           # dz
        arms = brs_action[4:16]           # 12D: left_arm(5) + right_arm(5) + grippers(2)
        
        # Build BigYM 16D action
        # Note: BigYM expects [dx, dy, dz, drz] order for floating base
        bigym_action = np.concatenate([
            mobile_base[:2],   # [0:2]  dx, dy
            torso,             # [2:3]  dz (torso)
            mobile_base[2:3],  # [3:4]  drz
            arms,              # [4:16] arms (left_arm 5 + right_arm 5 + grippers 2)
        ])
        
        return bigym_action

    def _obs_to_policy_input(
        self, 
        obs: Dict, 
        config: Dict, 
        device: torch.device
    ) -> Dict:
        """
        Convert BiGym observation to BRS policy input format.
        
        BigYM Native Proprioception (16D):
            - mobile_base_pos (3): [x, y, rz] from floating_base
            - torso (1): z position from floating_base[2]
            - left_arm (5): from qpos[0,1,2,3,12]
            - left_gripper (1): from grippers[0]
            - right_arm (5): from qpos[13,14,15,16,25]
            - right_gripper (1): from grippers[1]
        """
        # Initialize proprioception components (BigYM native: position-based)
        mobile_base_pos = np.zeros(3, dtype=np.float32)  # x, y, rz
        torso = np.zeros(1, dtype=np.float32)            # z
        left_arm = np.zeros(5, dtype=np.float32)
        left_gripper = np.zeros(1, dtype=np.float32)
        right_arm = np.zeros(5, dtype=np.float32)
        right_gripper = np.zeros(1, dtype=np.float32)
        
        # Extract mobile_base_pos and torso from floating_base position
        if 'proprioception_floating_base' in obs:
            fb = np.array(obs['proprioception_floating_base']).flatten()
            # floating_base: [x, y, z, rz]
            mobile_base_pos[0] = fb[0]  # x
            mobile_base_pos[1] = fb[1]  # y
            mobile_base_pos[2] = fb[3]  # rz
            torso[0] = fb[2]            # z
        
        # Extract arm joints from proprioception (qpos)
        if 'proprioception' in obs:
            qpos = np.array(obs['proprioception']).flatten()
            if len(qpos) >= 30:  # Full qpos available
                # Left arm: qpos[0,1,2,3,12]
                left_arm = np.array([qpos[0], qpos[1], qpos[2], qpos[3], qpos[12]])
                # Right arm: qpos[13,14,15,16,25]
                right_arm = np.array([qpos[13], qpos[14], qpos[15], qpos[16], qpos[25]])
        
        # Extract grippers
        if 'proprioception_grippers' in obs:
            grippers = np.array(obs['proprioception_grippers']).flatten()
            left_gripper[0] = grippers[0]
            right_gripper[0] = grippers[1]
        
        # Combine into 16D proprioception
        prop = np.concatenate([
            mobile_base_pos,  # 3: x, y, rz
            torso,            # 1: z
            left_arm,         # 5
            left_gripper,     # 1
            right_arm,        # 5
            right_gripper,    # 1
        ]).astype(np.float32)
        
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
            n_points = config.get('pcd_downsample_points', 4096)
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
            n_points = config.get('pcd_downsample_points', 4096)
            pcd_xyz = np.random.randn(n_points, 3).astype(np.float32) * 0.1
            pcd_rgb = np.random.rand(n_points, 3).astype(np.float32)
        
        # Normalize proprioception if stats available
        if config.get('normalize', False) and 'prop' in self._stats:
            prop = self._normalize_prop(prop)
        
        # Normalize PCD if stats available
        if config.get('normalize_pcd', False) and 'pcd' in self._stats:
            pcd_xyz = self._normalize_pcd(pcd_xyz)
        
        # Create policy input (BigYM Native format)
        num_obs = config.get('num_latest_obs', 2)
        policy_input = {
            'odom': {
                'mobile_base_pos': torch.from_numpy(prop[:3][None, None, :]).expand(1, num_obs, -1).to(device).float(),
            },
            'qpos': {
                'torso': torch.from_numpy(prop[3:4][None, None, :]).expand(1, num_obs, -1).to(device).float(),
                'left_arm': torch.from_numpy(prop[4:9][None, None, :]).expand(1, num_obs, -1).to(device).float(),
                'left_gripper': torch.from_numpy(prop[9:10][None, None, :]).expand(1, num_obs, -1).to(device).float(),
                'right_arm': torch.from_numpy(prop[10:15][None, None, :]).expand(1, num_obs, -1).to(device).float(),
                'right_gripper': torch.from_numpy(prop[15:16][None, None, :]).expand(1, num_obs, -1).to(device).float(),
            },
            'pointcloud': {
                'xyz': torch.from_numpy(pcd_xyz[None, None, :, :]).expand(1, num_obs, -1, -1).to(device).float(),
                'rgb': torch.from_numpy(pcd_rgb[None, None, :, :]).expand(1, num_obs, -1, -1).to(device).float(),
            }
        }
        
        return policy_input
    
    def _normalize_prop(self, prop: np.ndarray) -> np.ndarray:
        """Normalize proprioception to [-1, 1] using stats."""
        stats = self._stats['prop']
        
        # Use 'full' key directly if available (preferred for BigYM native)
        if 'full' in stats:
            prop_min = np.array(stats['full']['min'], dtype=np.float32)
            prop_max = np.array(stats['full']['max'], dtype=np.float32)
        else:
            # Build 16D bounds from components (legacy format)
            prop_min = []
            prop_max = []
            
            for key in ['mobile_base_pos', 'torso', 'arms']:
                if key not in stats:
                    continue
                val_min = stats[key]['min']
                val_max = stats[key]['max']
                if isinstance(val_min, list):
                    prop_min.extend(val_min)
                    prop_max.extend(val_max)
                else:
                    prop_min.append(val_min)
                    prop_max.append(val_max)
            
            prop_min = np.array(prop_min, dtype=np.float32)
            prop_max = np.array(prop_max, dtype=np.float32)
        
        # Normalize to [-1, 1]
        range_val = prop_max - prop_min
        range_val = np.where(range_val == 0, 1.0, range_val)
        normalized = 2.0 * (prop - prop_min) / range_val - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
    
    def _normalize_pcd(self, pcd_xyz: np.ndarray) -> np.ndarray:
        """Normalize PCD XYZ to [-1, 1] using stats."""
        stats = self._stats['pcd']
        pcd_min = np.array(stats['xyz']['min'], dtype=np.float32)
        pcd_max = np.array(stats['xyz']['max'], dtype=np.float32)
        
        range_val = pcd_max - pcd_min
        range_val = np.where(range_val == 0, 1.0, range_val)
        normalized = 2.0 * (pcd_xyz - pcd_min) / range_val - 1.0
        return np.clip(normalized, -1.0, 1.0).astype(np.float32)
    
    def _denormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """
        Denormalize actions from [-1, 1] back to original range.
        
        BRS 16D actions: [mobile_base(3), torso(1), arms(12)]
        """
        if 'action' not in self._stats:
            return actions
        
        stats = self._stats['action']
        
        # Check for 'full' key (preferred format)
        if 'full' in stats:
            action_min = np.array(stats['full']['min'], dtype=np.float32)
            action_max = np.array(stats['full']['max'], dtype=np.float32)
            range_val = action_max - action_min
            range_val = np.where(range_val == 0, 1.0, range_val)
            denormalized = (actions + 1.0) / 2.0 * range_val + action_min
            return denormalized.astype(np.float32)
        
        # Fallback to component-wise format
        mobile_base_act = actions[:3]
        torso_act = actions[3:4]
        arms_act = actions[4:16]
        
        denormalized_parts = []
        
        # Denormalize mobile_base
        if 'mobile_base' in stats and 'min' in stats['mobile_base']:
            mb_min = np.array(stats['mobile_base']['min'])
            mb_max = np.array(stats['mobile_base']['max'])
            range_val = mb_max - mb_min
            range_val = np.where(range_val == 0, 1.0, range_val)
            denorm_mb = (mobile_base_act + 1.0) / 2.0 * range_val + mb_min
            denormalized_parts.append(denorm_mb)
        else:
            denormalized_parts.append(mobile_base_act)
        
        # Denormalize torso
        if 'torso' in stats and 'min' in stats['torso']:
            t_min = np.array(stats['torso']['min'])
            t_max = np.array(stats['torso']['max'])
            range_val = t_max - t_min
            range_val = 1.0 if range_val == 0 else range_val
            denorm_t = (torso_act + 1.0) / 2.0 * range_val + t_min
            denormalized_parts.append(denorm_t)
        else:
            denormalized_parts.append(torso_act)
        
        # Denormalize arms
        if 'arms' in stats and 'min' in stats['arms']:
            a_min = np.array(stats['arms']['min'])
            a_max = np.array(stats['arms']['max'])
            range_val = a_max - a_min
            range_val = np.where(range_val == 0, 1.0, range_val)
            denorm_a = (arms_act + 1.0) / 2.0 * range_val + a_min
            denormalized_parts.append(denorm_a)
        else:
            denormalized_parts.append(arms_act)
        
        return np.concatenate(denormalized_parts, axis=-1).astype(np.float32)
    
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
                    
                    brs_action = np.concatenate(action_parts, axis=-1)  # 16D
                    
                    # Denormalize action
                    if config.get('normalize', False):
                        brs_action = self._denormalize_actions(brs_action)
                    
                    # Convert BRS 16D action to BigYM 16D action
                    # BRS 16D: [mobile_base(3), torso(1), arms(12)]
                    # BigYM 16D: [floating_base(4), arms(12)]
                    action = self._brs_to_bigym_action(brs_action)
                    
                    # CRITICAL: Apply decimation for 500Hz/50Hz frequency mismatch
                    # Policy outputs 50Hz actions, but env runs at 500Hz
                    # Floating base deltas are SUM of 10 x 500Hz deltas in demo
                    # So we divide floating base by decimation and execute decimation env steps
                    decimation = ENV_DECIMATION
                    action_500hz = action.copy()
                    action_500hz[0:4] = action[0:4] / decimation  # Divide floating base delta
                    
                    # Execute decimation env steps
                    step_reward = 0.0
                    for _ in range(decimation):
                        # Clip action to environment action space
                        action_clipped = np.clip(action_500hz, self._env.action_space.low, self._env.action_space.high)
                        
                        # Execute action
                        obs, reward, terminated, truncated, info = self._env.step(action_clipped)
                        step_reward += reward
                        
                        if terminated or truncated:
                            break
                    
                    episode_reward += step_reward
                    
                    # Record frame (only once per policy step to keep video manageable)
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

