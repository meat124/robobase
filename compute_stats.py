#!/usr/bin/env python3
"""
Compute normalization statistics for BRS policy training.

Computes min/max statistics for:
1. Actions (16D): mobile_base(3) + torso(1) + arms(12)
2. Proprioception (16D): mobile_base_vel(3) + torso(1) + arms(12)
3. PCD XYZ coordinates (3D)

Saves to JSON files in the same directory as the HDF5 file.
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import json
import argparse
from tqdm import tqdm

# Add robobase to path
sys.path.insert(0, str(Path(__file__).parent))


def compute_stats(hdf5_path: str, output_dir: str = None, pcd_root: str = None):
    """Compute normalization statistics from HDF5 demos."""
    
    hdf5_path = Path(hdf5_path)
    if output_dir is None:
        output_dir = hdf5_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"="*80)
    print(f"Computing Normalization Statistics")
    print(f"="*80)
    print(f"HDF5: {hdf5_path}")
    print(f"Output: {output_dir}")
    
    # Collect data
    all_actions = []
    all_mobile_base_vel = []
    all_torso = []
    all_left_arm = []
    all_left_gripper = []
    all_right_arm = []
    all_right_gripper = []
    all_pcd_xyz = []
    
    dt = 0.02  # 50Hz
    
    with h5py.File(hdf5_path, 'r') as f:
        # Detect structure
        if 'data' in f:
            demo_keys = sorted([k for k in f['data'].keys() if k.startswith('demo_')])
            get_demo = lambda k: f['data'][k]
        else:
            demo_keys = sorted([k for k in f.keys() if k.startswith('demo_')])
            get_demo = lambda k: f[k]
        
        print(f"\nProcessing {len(demo_keys)} demos...")
        
        for demo_key in tqdm(demo_keys, desc="Processing demos"):
            demo = get_demo(demo_key)
            
            # Actions (17D -> 16D)
            actions = demo['actions'][()]
            # 17D: [fb_x, fb_y, fb_z, fb_rz, torso, arms(12)]
            # 16D: [fb_x_vel, fb_y_vel, fb_rz_vel, torso_z, arms(12)]
            
            # Convert floating base deltas to velocities
            fb_vel = actions[:, [0, 1, 3]] / dt  # x, y, rz velocities
            
            # Torso: accumulate deltas to get absolute positions
            torso_delta = actions[:, 2]
            
            # Get initial torso position from proprioception
            if 'obs' in demo and 'proprioception_floating_base' in demo['obs']:
                prop_fb = demo['obs']['proprioception_floating_base'][()]
                initial_torso = prop_fb[0, 2]
            else:
                prop = demo['proprioception'][()]
                initial_torso = prop[0, 29]  # z at index 29 in 62D prop
            
            torso_abs = initial_torso + np.cumsum(torso_delta)
            
            # Arms (12D)
            arms = actions[:, 4:17]
            if arms.shape[1] == 13:
                arms = arms[:, :12]  # Take first 12
            
            # Combine for action stats
            action_16d = np.concatenate([fb_vel, torso_abs.reshape(-1, 1), arms], axis=1)
            all_actions.append(action_16d)
            
            # Proprioception
            if 'obs' in demo:
                prop = demo['obs']['proprioception'][()]
                prop_fb = demo['obs']['proprioception_floating_base'][()]
                if 'proprioception_grippers' in demo['obs']:
                    prop_grippers = demo['obs']['proprioception_grippers'][()]
                else:
                    prop_grippers = np.zeros((len(prop), 2), dtype=np.float32)
            else:
                prop = demo['proprioception'][()]
                prop_fb = prop[:, 27:31]  # x, y, z, rz
                prop_grippers = np.zeros((len(prop), 2), dtype=np.float32)
            
            # Compute mobile base velocity from position deltas
            mobile_base_vel = np.zeros((len(prop_fb), 3), dtype=np.float32)
            for i in range(1, len(prop_fb)):
                mobile_base_vel[i] = (prop_fb[i, [0, 1, 3]] - prop_fb[i-1, [0, 1, 3]]) / dt
            
            # Extract arm joints
            left_arm = np.concatenate([prop[:, 0:4], prop[:, 12:13]], axis=-1)  # 5D
            right_arm = np.concatenate([prop[:, 13:17], prop[:, 25:26]], axis=-1)  # 5D
            
            all_mobile_base_vel.append(mobile_base_vel)
            all_torso.append(prop_fb[:, 2:3])  # z position
            all_left_arm.append(left_arm)
            all_left_gripper.append(prop_grippers[:, 0:1])
            all_right_arm.append(right_arm)
            all_right_gripper.append(prop_grippers[:, 1:2])
    
    # Concatenate all data
    all_actions = np.concatenate(all_actions, axis=0)
    all_mobile_base_vel = np.concatenate(all_mobile_base_vel, axis=0)
    all_torso = np.concatenate(all_torso, axis=0)
    all_left_arm = np.concatenate(all_left_arm, axis=0)
    all_left_gripper = np.concatenate(all_left_gripper, axis=0)
    all_right_arm = np.concatenate(all_right_arm, axis=0)
    all_right_gripper = np.concatenate(all_right_gripper, axis=0)
    
    print(f"\nData shapes:")
    print(f"  Actions: {all_actions.shape}")
    print(f"  Mobile base vel: {all_mobile_base_vel.shape}")
    print(f"  Torso: {all_torso.shape}")
    print(f"  Left arm: {all_left_arm.shape}")
    
    # Compute action statistics
    action_stats = {
        'mobile_base': {
            'min': all_actions[:, 0:3].min(axis=0).tolist(),
            'max': all_actions[:, 0:3].max(axis=0).tolist(),
            'mean': all_actions[:, 0:3].mean(axis=0).tolist(),
            'std': all_actions[:, 0:3].std(axis=0).tolist(),
        },
        'torso': {
            'min': float(all_actions[:, 3].min()),
            'max': float(all_actions[:, 3].max()),
            'mean': float(all_actions[:, 3].mean()),
            'std': float(all_actions[:, 3].std()),
        },
        'arms': {
            'min': all_actions[:, 4:16].min(axis=0).tolist(),
            'max': all_actions[:, 4:16].max(axis=0).tolist(),
            'mean': all_actions[:, 4:16].mean(axis=0).tolist(),
            'std': all_actions[:, 4:16].std(axis=0).tolist(),
        }
    }
    
    # Compute proprioception statistics
    prop_stats = {
        'mobile_base_vel': {
            'min': all_mobile_base_vel.min(axis=0).tolist(),
            'max': all_mobile_base_vel.max(axis=0).tolist(),
            'mean': all_mobile_base_vel.mean(axis=0).tolist(),
            'std': all_mobile_base_vel.std(axis=0).tolist(),
        },
        'torso': {
            'min': float(all_torso.min()),
            'max': float(all_torso.max()),
            'mean': float(all_torso.mean()),
            'std': float(all_torso.std()),
        },
        'left_arm': {
            'min': all_left_arm.min(axis=0).tolist(),
            'max': all_left_arm.max(axis=0).tolist(),
            'mean': all_left_arm.mean(axis=0).tolist(),
            'std': all_left_arm.std(axis=0).tolist(),
        },
        'left_gripper': {
            'min': float(all_left_gripper.min()),
            'max': float(all_left_gripper.max()),
            'mean': float(all_left_gripper.mean()),
            'std': float(all_left_gripper.std()),
        },
        'right_arm': {
            'min': all_right_arm.min(axis=0).tolist(),
            'max': all_right_arm.max(axis=0).tolist(),
            'mean': all_right_arm.mean(axis=0).tolist(),
            'std': all_right_arm.std(axis=0).tolist(),
        },
        'right_gripper': {
            'min': float(all_right_gripper.min()),
            'max': float(all_right_gripper.max()),
            'mean': float(all_right_gripper.mean()),
            'std': float(all_right_gripper.std()),
        },
    }
    
    # PCD statistics (if available)
    pcd_stats = None
    if pcd_root:
        pcd_root = Path(pcd_root)
        if pcd_root.exists():
            print(f"\nProcessing PCD files from {pcd_root}...")
            all_xyz = []
            for demo_dir in tqdm(list(pcd_root.iterdir()), desc="Loading PCDs"):
                if demo_dir.is_dir() and demo_dir.name.startswith('demo_'):
                    for camera_dir in demo_dir.iterdir():
                        if camera_dir.is_dir():
                            for npz_file in list(camera_dir.glob('*.npz'))[:10]:  # Sample
                                data = np.load(npz_file)
                                all_xyz.append(data['xyz'])
            
            if len(all_xyz) > 0:
                all_xyz = np.concatenate(all_xyz, axis=0)
                pcd_stats = {
                    'xyz': {
                        'min': all_xyz.min(axis=0).tolist(),
                        'max': all_xyz.max(axis=0).tolist(),
                        'mean': all_xyz.mean(axis=0).tolist(),
                        'std': all_xyz.std(axis=0).tolist(),
                    }
                }
    
    # If no PCD stats, use default reasonable bounds
    if pcd_stats is None:
        pcd_stats = {
            'xyz': {
                'min': [-2.0, -2.0, 0.0],
                'max': [2.0, 2.0, 2.0],
                'mean': [0.0, 0.0, 1.0],
                'std': [1.0, 1.0, 0.5],
            }
        }
        print(f"\nUsing default PCD stats (no PCD files found)")
    
    # Save statistics
    action_stats_path = output_dir / "action_stats.json"
    prop_stats_path = output_dir / "prop_stats.json"
    pcd_stats_path = output_dir / "pcd_stats.json"
    
    with open(action_stats_path, 'w') as f:
        json.dump(action_stats, f, indent=2)
    print(f"\nSaved action stats to: {action_stats_path}")
    
    with open(prop_stats_path, 'w') as f:
        json.dump(prop_stats, f, indent=2)
    print(f"Saved prop stats to: {prop_stats_path}")
    
    with open(pcd_stats_path, 'w') as f:
        json.dump(pcd_stats, f, indent=2)
    print(f"Saved PCD stats to: {pcd_stats_path}")
    
    # Print summary
    print(f"\n" + "="*80)
    print(f"Statistics Summary")
    print(f"="*80)
    
    print(f"\nAction Stats (16D):")
    print(f"  mobile_base (3D): min={action_stats['mobile_base']['min']}")
    print(f"                    max={action_stats['mobile_base']['max']}")
    print(f"  torso (1D): min={action_stats['torso']['min']:.4f}, max={action_stats['torso']['max']:.4f}")
    print(f"  arms (12D): min range=[{min(action_stats['arms']['min']):.4f}, {max(action_stats['arms']['min']):.4f}]")
    print(f"              max range=[{min(action_stats['arms']['max']):.4f}, {max(action_stats['arms']['max']):.4f}]")
    
    print(f"\nProprioception Stats (16D):")
    print(f"  mobile_base_vel (3D): min={prop_stats['mobile_base_vel']['min']}")
    print(f"                        max={prop_stats['mobile_base_vel']['max']}")
    print(f"  torso (1D): min={prop_stats['torso']['min']:.4f}, max={prop_stats['torso']['max']:.4f}")
    
    print(f"\nPCD Stats:")
    print(f"  xyz min: {pcd_stats['xyz']['min']}")
    print(f"  xyz max: {pcd_stats['xyz']['max']}")
    
    return action_stats, prop_stats, pcd_stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute normalization statistics")
    parser.add_argument("--hdf5", type=str, 
                        default="/home/hyunjin/bigym_ws/data/demonstrations/saucepan_to_hob/demos.hdf5",
                        help="HDF5 file path")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--pcd-root", type=str, default=None, help="PCD root directory")
    args = parser.parse_args()
    
    compute_stats(args.hdf5, args.output, args.pcd_root)
