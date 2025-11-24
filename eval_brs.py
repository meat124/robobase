"""
Evaluate trained BRS Policy in BigYM environment and save videos
"""

import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import argparse

# Add brs-algo to path
brs_algo_path = Path(__file__).parent.parent / "brs-algo"
sys.path.insert(0, str(brs_algo_path))

# Add bigym to path
bigym_path = Path(__file__).parent.parent / "bigym"
sys.path.insert(0, str(bigym_path))

from brs_algo.learning.policy import WBVIMAPolicy
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

# Import BiGym
from bigym.envs.pick_and_place import SaucepanToHob
from bigym.action_modes import JointPositionActionMode
from bigym.robots.configs.h1 import H1

# Import video recording
from robobase.video import VideoRecorder


def load_normalization_stats(config: Dict) -> Dict:
    """Load normalization statistics for actions and observations"""
    stats = {}
    
    if config.get('normalize', False):
        # Load action stats
        action_stats_path = Path(config.get('action_stats_path', ''))
        if action_stats_path.exists():
            import json
            with open(action_stats_path, 'r') as f:
                stats['action'] = json.load(f)
            print(f"✓ Loaded action stats from {action_stats_path}")
        
        # Load prop stats
        prop_stats_path = Path(config.get('prop_stats_path', ''))
        if prop_stats_path.exists():
            import json
            with open(prop_stats_path, 'r') as f:
                stats['prop'] = json.load(f)
            print(f"✓ Loaded prop stats from {prop_stats_path}")
        
        # Load PCD stats
        pcd_stats_path = Path(config.get('pcd_stats_path', ''))
        if pcd_stats_path.exists():
            import json
            with open(pcd_stats_path, 'r') as f:
                stats['pcd'] = json.load(f)
            print(f"✓ Loaded PCD stats from {pcd_stats_path}")
    
    return stats


def normalize_data(data: np.ndarray, stats: Dict, key: str) -> np.ndarray:
    """Normalize data to [-1, 1] using min-max normalization"""
    if key not in stats:
        return data
    
    min_val = np.array(stats[key]['min'])
    max_val = np.array(stats[key]['max'])
    
    # Min-max normalization to [-1, 1]
    normalized = 2.0 * (data - min_val) / (max_val - min_val + 1e-8) - 1.0
    return np.clip(normalized, -1.0, 1.0)


def denormalize_actions(actions: np.ndarray, stats: Dict) -> np.ndarray:
    """Denormalize actions from [-1, 1] back to original range
    
    action_stats.json has keys: mobile_base, torso, arms (not a single 'action' key)
    actions array is concatenated: [mobile_base(3) + torso(1) + arms(12)] = 16
    """
    # Extract individual components based on action structure
    mobile_base_act = actions[:3]  # 3 dims
    torso_act = actions[3:4]  # 1 dim
    arms_act = actions[4:16]  # 12 dims
    
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


def create_bigym_env(config: Dict):
    """Create BiGym environment"""
    # Create SaucepanToHob environment with H1 robot and JointPosition action mode (absolute)
    # This matches the training data configuration
    env = SaucepanToHob(
        action_mode=JointPositionActionMode(floating_base=True, absolute=True),
        robot_cls=H1,
        render_mode='rgb_array',
    )
    
    return env


def obs_to_policy_input(obs: Dict, config: Dict, stats: Dict, device: torch.device) -> Dict:
    """Convert BiGym observation to BRS policy input format"""
    
    # Extract proprioception from BiGym observation
    # BiGym uses: proprioception, proprioception_grippers, proprioception_floating_base
    prop_data = []
    
    # mobile_base_vel (3): from proprioception_floating_base_actions
    if 'proprioception_floating_base_actions' in obs:
        base_vel = obs['proprioception_floating_base_actions']
        if not isinstance(base_vel, np.ndarray):
            base_vel = np.array(base_vel)
        prop_data.append(base_vel.flatten()[:3])  # Take first 3 values
    
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
        print("Warning: No proprioception data found, using zeros")
        prop = np.zeros(config['prop_dim'], dtype=np.float32)
    else:
        prop = np.concatenate(prop_data, axis=-1).astype(np.float32)
        # Ensure correct size
        if prop.shape[0] != config['prop_dim']:
            print(f"Warning: proprioception size mismatch (got {prop.shape[0]}, expected {config['prop_dim']}), padding/truncating")
            if prop.shape[0] < config['prop_dim']:
                prop = np.pad(prop, (0, config['prop_dim'] - prop.shape[0]))
            else:
                prop = prop[:config['prop_dim']]
    
    # Normalize proprioception
    if config.get('normalize', False) and 'prop' in stats:
        # prop_stats has individual keys (mobile_base_vel, torso, etc) not a single 'prop' key
        # So we skip normalization for now or need to normalize each component separately
        pass  # Skip normalization since we truncated/padded the prop vector
    
    # Extract point cloud (if available)
    # For now, use dummy PCD - you'll need to implement PCD extraction from BiGym
    n_points = config.get('pcd_downsample_points', 2048)
    pcd_xyz = np.random.randn(n_points, 3).astype(np.float32) * 0.1
    pcd_rgb = np.random.rand(n_points, 3).astype(np.float32)
    
    # Create policy input - BRS expects (B, T, ...) format matching prop_keys structure
    # prop_keys: ["odom/base_velocity", "qpos/torso", "qpos/left_arm", "qpos/left_gripper", "qpos/right_arm", "qpos/right_gripper"]
    # B=1 (batch), T=num_latest_obs (time/observation history)
    num_obs = config.get('num_latest_obs', 1)
    policy_input = {
        'odom': {
            'base_velocity': torch.from_numpy(prop[:3][None, None, :]).expand(1, num_obs, -1).to(device),  # [1, T, 3]
        },
        'qpos': {
            'torso': torch.from_numpy(prop[3:4][None, None, :]).expand(1, num_obs, -1).to(device),  # [1, T, 1]
            'left_arm': torch.from_numpy(prop[4:9][None, None, :]).expand(1, num_obs, -1).to(device),  # [1, T, 5]
            'left_gripper': torch.from_numpy(prop[9:10][None, None, :]).expand(1, num_obs, -1).to(device),  # [1, T, 1]
            'right_arm': torch.from_numpy(prop[10:15][None, None, :]).expand(1, num_obs, -1).to(device),  # [1, T, 5]
            'right_gripper': torch.from_numpy(prop[15:16][None, None, :]).expand(1, num_obs, -1).to(device),  # [1, T, 1]
        },
        'pointcloud': {
            'xyz': torch.from_numpy(pcd_xyz[None, None, :, :]).expand(1, num_obs, -1, -1).to(device),  # [1, T, n_points, 3]
            'rgb': torch.from_numpy(pcd_rgb[None, None, :, :]).expand(1, num_obs, -1, -1).to(device),  # [1, T, n_points, 3]
        }
    }
    
    return policy_input


def evaluate_policy(
    checkpoint_path: str,
    config_path: str,
    num_episodes: int = 5,
    save_video: bool = True,
    output_dir: Optional[str] = None,
):
    """Evaluate BRS policy in BiGym environment"""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load normalization stats
    stats = load_normalization_stats(config)
    
    # Create environment
    print("Creating BiGym environment...")
    env = create_bigym_env(config)
    
    # Load policy
    print(f"Loading policy from {checkpoint_path}...")
    
    # Create noise scheduler
    noise_scheduler = DDIMScheduler(**config['noise_scheduler'])
    
    # Initialize policy
    policy = WBVIMAPolicy(
        num_latest_obs=config['num_latest_obs'],
        action_keys=config['action_keys'],
        action_key_dims=config['action_key_dims'],
        action_prediction_horizon=config['action_prediction_horizon'],
        prop_keys=config['prop_keys'],
        prop_dim=config['prop_dim'],
        action_dim=config['action_dim'],
        num_denoise_steps_per_inference=config.get('num_denoise_steps_per_inference', 16),
        use_modality_type_tokens=config.get('use_modality_type_tokens', False),
        prop_mlp_hidden_depth=config.get('prop_mlp_hidden_depth', 2),
        prop_mlp_hidden_dim=config.get('prop_mlp_hidden_dim', 256),
        pointnet_n_coordinates=config.get('pointnet_n_coordinates', 3),
        pointnet_n_color=config.get('pointnet_n_color', 3),
        pointnet_hidden_depth=config.get('pointnet_hidden_depth', 2),
        pointnet_hidden_dim=config.get('pointnet_hidden_dim', 256),
        xf_n_embd=config.get('xf_n_embd', 256),
        xf_n_layer=config.get('xf_n_layer', 2),
        xf_n_head=config.get('xf_n_head', 8),
        xf_dropout_rate=config.get('xf_dropout_rate', 0.1),
        xf_use_geglu=config.get('xf_use_geglu', True),
        learnable_action_readout_token=config.get('learnable_action_readout_token', False),
        diffusion_step_embed_dim=config.get('diffusion_step_embed_dim', 128),
        unet_down_dims=config.get('unet_down_dims', [64, 128]),
        unet_kernel_size=config.get('unet_kernel_size', 5),
        unet_n_groups=config.get('unet_n_groups', 8),
        unet_cond_predict_scale=config.get('unet_cond_predict_scale', True),
        noise_scheduler=noise_scheduler,
    ).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if 'state_dict' in checkpoint:
        # Lightning checkpoint
        state_dict = {k.replace('policy.', ''): v for k, v in checkpoint['state_dict'].items() if k.startswith('policy.')}
        policy.load_state_dict(state_dict)
    else:
        policy.load_state_dict(checkpoint)
    
    policy.eval()
    print("✓ Policy loaded successfully")
    
    # Setup video recording
    if output_dir is None:
        output_dir = Path(checkpoint_path).parent.parent / "eval_videos"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    video_recorder = VideoRecorder(output_dir if save_video else None)
    
    # Run evaluation episodes
    print(f"\nRunning {num_episodes} evaluation episodes...")
    successes = 0
    total_rewards = 0
    total_steps = 0
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        video_recorder.init(env, enabled=save_video)
        
        done = False
        episode_reward = 0
        episode_steps = 0
        
        # Observation history for multi-observation input
        obs_history = [obs] * config['num_latest_obs']
        
        while not done:
            # Prepare policy input
            policy_input = obs_to_policy_input(obs, config, stats, device)
            
            # Get action from policy - use act() method instead of forward()
            with torch.no_grad():
                action_chunks = policy.act(policy_input)
            
            # Debug: check output (on first step of first episode)
            if episode_steps == 0:
                print(f"\n=== Debug Episode {episode+1}, Step {episode_steps} ===")
                print(f"action_chunks type: {type(action_chunks)}")
                if isinstance(action_chunks, dict):
                    print(f"action_chunks keys: {list(action_chunks.keys())}")
                    for k in action_chunks.keys():
                        v = action_chunks[k]
                        print(f"  '{k}': shape={v.shape}, dtype={v.dtype}")
                elif isinstance(action_chunks, torch.Tensor):
                    print(f"action_chunks is a TENSOR with shape: {action_chunks.shape}")
                else:
                    print(f"action_chunks unexpected type!")
                print("="*50 + "\n")
            
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
            if config.get('normalize', False) and stats.get('action'):
                action = denormalize_actions(action, stats['action'])
            
            # Clip action to environment action space
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            # Execute action
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            
            video_recorder.record(env)
        
        # Save video
        if save_video:
            video_name = f"episode_{episode+1}.mp4"
            video_recorder.save(video_name)
            print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, Steps={episode_steps}, Video saved to {output_dir / video_name}")
        else:
            print(f"  Episode {episode+1}: Reward={episode_reward:.2f}, Steps={episode_steps}")
        
        # Check success
        if info.get('task_success', False):
            successes += 1
        
        total_rewards += episode_reward
        total_steps += episode_steps
    
    # Print summary
    print("\n" + "="*80)
    print("Evaluation Summary:")
    print("="*80)
    print(f"Success Rate: {successes}/{num_episodes} ({100*successes/num_episodes:.1f}%)")
    print(f"Average Reward: {total_rewards/num_episodes:.2f}")
    print(f"Average Steps: {total_steps/num_episodes:.1f}")
    if save_video:
        print(f"Videos saved to: {output_dir}")
    print("="*80)
    
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate BRS Policy')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--config', type=str, default='robobase/cfgs/brs_config.yaml', help='Path to config file')
    parser.add_argument('--num-episodes', type=int, default=5, help='Number of evaluation episodes')
    parser.add_argument('--no-video', action='store_true', help='Disable video recording')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory for videos')
    
    args = parser.parse_args()
    
    evaluate_policy(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        num_episodes=args.num_episodes,
        save_video=not args.no_video,
        output_dir=args.output_dir,
    )
