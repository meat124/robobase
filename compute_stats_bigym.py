#!/usr/bin/env python3
"""
Compute normalization statistics for BRS policy training with BigYM Native Format.

Uses RAW BigYM format, reordered for BRS 3-part autoregressive:
- Actions (16D): mobile_base(3) + torso(1) + arms(12)
  - mobile_base: (dx, dy, drz) - DELTA mode
  - torso: (dz) - DELTA mode
  - arms: left_arm(5) + right_arm(5) + grippers(2) - ABSOLUTE

- Proprioception (16D): mobile_base_pos(3) + torso_z(1) + arms(12)
  - mobile_base_pos: (x, y, rz) - ABSOLUTE positions
  - torso_z: (z) - ABSOLUTE position
  - arms: left_arm(5) + right_arm(5) + grippers(2) - ABSOLUTE

Saves to JSON files in the same directory as the HDF5 file.
"""

import sys
from pathlib import Path
import numpy as np
import h5py
import json
import argparse
from tqdm import tqdm


def compute_stats_bigym(hdf5_path: str, output_dir: str = None, pcd_root: str = None):
    """Compute normalization statistics from HDF5 demos for BigYM native format."""
    
    hdf5_path = Path(hdf5_path)
    if output_dir is None:
        output_dir = hdf5_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"="*80)
    print(f"Computing BigYM Native Format Statistics (3-part)")
    print(f"="*80)
    print(f"HDF5: {hdf5_path}")
    print(f"Output: {output_dir}")
    
    # Collect data for actions
    all_mobile_base_delta = []  # (dx, dy, drz) from actions
    all_torso_delta = []  # (dz) from actions
    all_arms = []  # left_arm(5) + right_arm(5) + grippers(2) from actions
    
    # Collect data for proprioception
    all_mobile_base_pos = []  # (x, y, rz) from proprioception
    all_torso_z = []  # (z) from proprioception
    all_prop_arms = []  # left_arm(5) + right_arm(5) + grippers(2) from proprioception
    
    all_pcd_xyz = []
    
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
            
            # Actions (16D or 17D from BigYM)
            actions = demo['actions'][()]
            T = len(actions)
            
            # Extract for BRS 3-part structure:
            # BigYM raw: [dx, dy, dz, drz, left_arm(5), right_arm(5), grippers(2)]
            # BRS: mobile_base(dx,dy,drz) + torso(dz) + arms(12)
            mobile_base_delta = actions[:, [0, 1, 3]]  # (dx, dy, drz) - 3D
            torso_delta = actions[:, 2:3]  # (dz) - 1D
            all_mobile_base_delta.append(mobile_base_delta)
            all_torso_delta.append(torso_delta)
            
            # Extract arms (12D): left_arm(5) + right_arm(5) + grippers(2)
            # From BigYM 17D: indices 5-16 (skip index 4 which is redundant for grippers)
            if actions.shape[1] == 17:
                # Original 17D: skip index 4 (it's the old torso/gripper)
                arms = actions[:, 5:17]  # 12D
            elif actions.shape[1] == 16:
                # Already 16D: floating_base(4) + arms(12)
                arms = actions[:, 4:16]  # 12D
            else:
                raise ValueError(f"Unexpected action dimension: {actions.shape[1]}")
            
            all_arms.append(arms)
            
            # Proprioception
            if 'obs' in demo:
                prop = demo['obs']['proprioception'][()]  # 26D qpos
                prop_fb = demo['obs']['proprioception_floating_base'][()]  # 4D (x, y, z, rz)
                if 'proprioception_grippers' in demo['obs']:
                    prop_grippers = demo['obs']['proprioception_grippers'][()]  # 2D
                else:
                    prop_grippers = np.zeros((T+1, 2), dtype=np.float32)
            elif 'proprioception_floating_base' in demo:
                prop = demo['proprioception'][()]  # 60D
                prop_fb = demo['proprioception_floating_base'][()]  # 4D (x, y, z, rz)
                if 'proprioception_grippers' in demo:
                    prop_grippers = demo['proprioception_grippers'][()]  # 2D
                else:
                    prop_grippers = np.zeros((T+1, 2), dtype=np.float32)
            else:
                # Legacy format
                full_prop = demo['proprioception'][()]  # 62D
                prop_fb = full_prop[:, 27:31]  # x, y, z, rz
                prop_grippers = np.zeros((T+1, 2), dtype=np.float32)
                prop = full_prop[:, :26]  # qpos
            
            # mobile_base_pos: [x, y, rz] from floating_base
            mobile_base_pos = prop_fb[:, [0, 1, 3]]  # 3D
            all_mobile_base_pos.append(mobile_base_pos)
            
            # torso_z: [z] from floating_base
            torso_z = prop_fb[:, 2:3]  # 1D
            all_torso_z.append(torso_z)
            
            # Extract arm proprioception (12D):
            # left_arm: qpos[0:4] + qpos[12] = 5D
            # right_arm: qpos[13:17] + qpos[25] = 5D
            # grippers: 2D
            left_arm_prop = np.concatenate([prop[:, 0:4], prop[:, 12:13]], axis=-1)  # 5D
            right_arm_prop = np.concatenate([prop[:, 13:17], prop[:, 25:26]], axis=-1)  # 5D
            arms_prop = np.concatenate([left_arm_prop, right_arm_prop, prop_grippers], axis=-1)  # 12D
            all_prop_arms.append(arms_prop)
    
    # Concatenate all data
    all_mobile_base_delta = np.concatenate(all_mobile_base_delta, axis=0)
    all_torso_delta = np.concatenate(all_torso_delta, axis=0)
    all_arms = np.concatenate(all_arms, axis=0)
    all_mobile_base_pos = np.concatenate(all_mobile_base_pos, axis=0)
    all_torso_z = np.concatenate(all_torso_z, axis=0)
    all_prop_arms = np.concatenate(all_prop_arms, axis=0)
    
    print(f"\nData shapes:")
    print(f"  Mobile base delta (actions): {all_mobile_base_delta.shape}")
    print(f"  Torso delta (actions): {all_torso_delta.shape}")
    print(f"  Arms (actions): {all_arms.shape}")
    print(f"  Mobile base pos (prop): {all_mobile_base_pos.shape}")
    print(f"  Torso z (prop): {all_torso_z.shape}")
    print(f"  Arms (prop): {all_prop_arms.shape}")
    
    # Compute action statistics for BRS 3-part policy
    # BRS structure: mobile_base(3) -> torso(1) -> arms(12)
    action_stats = {
        'mobile_base': {
            'min': all_mobile_base_delta.min(axis=0).tolist(),
            'max': all_mobile_base_delta.max(axis=0).tolist(),
            'mean': all_mobile_base_delta.mean(axis=0).tolist(),
            'std': all_mobile_base_delta.std(axis=0).tolist(),
        },
        'torso': {
            'min': all_torso_delta.min(axis=0).tolist(),
            'max': all_torso_delta.max(axis=0).tolist(),
            'mean': all_torso_delta.mean(axis=0).tolist(),
            'std': all_torso_delta.std(axis=0).tolist(),
        },
        'arms': {
            'min': all_arms.min(axis=0).tolist(),
            'max': all_arms.max(axis=0).tolist(),
            'mean': all_arms.mean(axis=0).tolist(),
            'std': all_arms.std(axis=0).tolist(),
        },
        # Also include 'full' for convenience (16D)
        'full': {
            'min': np.concatenate([all_mobile_base_delta, all_torso_delta, all_arms], axis=1).min(axis=0).tolist(),
            'max': np.concatenate([all_mobile_base_delta, all_torso_delta, all_arms], axis=1).max(axis=0).tolist(),
            'mean': np.concatenate([all_mobile_base_delta, all_torso_delta, all_arms], axis=1).mean(axis=0).tolist(),
            'std': np.concatenate([all_mobile_base_delta, all_torso_delta, all_arms], axis=1).std(axis=0).tolist(),
        }
    }
    
    # Compute proprioception statistics
    # Same structure: mobile_base_pos(3) + torso_z(1) + arms(12)
    prop_stats = {
        'mobile_base_pos': {
            'min': all_mobile_base_pos.min(axis=0).tolist(),
            'max': all_mobile_base_pos.max(axis=0).tolist(),
            'mean': all_mobile_base_pos.mean(axis=0).tolist(),
            'std': all_mobile_base_pos.std(axis=0).tolist(),
        },
        'torso': {
            'min': all_torso_z.min(axis=0).tolist(),
            'max': all_torso_z.max(axis=0).tolist(),
            'mean': all_torso_z.mean(axis=0).tolist(),
            'std': all_torso_z.std(axis=0).tolist(),
        },
        'arms': {
            'min': all_prop_arms.min(axis=0).tolist(),
            'max': all_prop_arms.max(axis=0).tolist(),
            'mean': all_prop_arms.mean(axis=0).tolist(),
            'std': all_prop_arms.std(axis=0).tolist(),
        },
        # Also include 'full' for convenience (16D)
        'full': {
            'min': np.concatenate([all_mobile_base_pos, all_torso_z, all_prop_arms], axis=1).min(axis=0).tolist(),
            'max': np.concatenate([all_mobile_base_pos, all_torso_z, all_prop_arms], axis=1).max(axis=0).tolist(),
            'mean': np.concatenate([all_mobile_base_pos, all_torso_z, all_prop_arms], axis=1).mean(axis=0).tolist(),
            'std': np.concatenate([all_mobile_base_pos, all_torso_z, all_prop_arms], axis=1).std(axis=0).tolist(),
        }
    }
    
    # Save statistics
    action_stats_path = output_dir / "action_stats.json"
    with open(action_stats_path, 'w') as f:
        json.dump(action_stats, f, indent=2)
    print(f"\n✓ Saved action stats to: {action_stats_path}")
    
    prop_stats_path = output_dir / "prop_stats.json"
    with open(prop_stats_path, 'w') as f:
        json.dump(prop_stats, f, indent=2)
    print(f"✓ Saved prop stats to: {prop_stats_path}")
    
    # Print statistics
    print(f"\n" + "="*80)
    print("ACTION STATISTICS (BRS 3-part from BigYM)")
    print("="*80)
    print(f"\nMobile Base Delta (3D: dx, dy, drz):")
    print(f"  Min: {[f'{x:.6f}' for x in action_stats['mobile_base']['min']]}")
    print(f"  Max: {[f'{x:.6f}' for x in action_stats['mobile_base']['max']]}")
    print(f"  Mean: {[f'{x:.6f}' for x in action_stats['mobile_base']['mean']]}")
    print(f"  Std: {[f'{x:.6f}' for x in action_stats['mobile_base']['std']]}")
    
    print(f"\nTorso Delta (1D: dz):")
    print(f"  Min: {[f'{x:.6f}' for x in action_stats['torso']['min']]}")
    print(f"  Max: {[f'{x:.6f}' for x in action_stats['torso']['max']]}")
    
    print(f"\nArms (12D: left_arm(5) + right_arm(5) + grippers(2)):")
    print(f"  Min: {[f'{x:.4f}' for x in action_stats['arms']['min']]}")
    print(f"  Max: {[f'{x:.4f}' for x in action_stats['arms']['max']]}")
    
    print(f"\n" + "="*80)
    print("PROPRIOCEPTION STATISTICS (BRS 3-part from BigYM)")
    print("="*80)
    print(f"\nMobile Base Position (3D: x, y, rz):")
    print(f"  Min: {[f'{x:.4f}' for x in prop_stats['mobile_base_pos']['min']]}")
    print(f"  Max: {[f'{x:.4f}' for x in prop_stats['mobile_base_pos']['max']]}")
    
    print(f"\nTorso Z (1D):")
    print(f"  Min: {[f'{x:.4f}' for x in prop_stats['torso']['min']]}")
    print(f"  Max: {[f'{x:.4f}' for x in prop_stats['torso']['max']]}")
    
    print(f"\nArms Proprioception (12D):")
    print(f"  Min: {[f'{x:.4f}' for x in prop_stats['arms']['min']]}")
    print(f"  Max: {[f'{x:.4f}' for x in prop_stats['arms']['max']]}")
    
    # Compute PCD statistics if pcd_root is provided
    if pcd_root is not None:
        pcd_root = Path(pcd_root)
        if pcd_root.exists():
            print(f"\n" + "="*80)
            print("COMPUTING PCD STATISTICS")
            print("="*80)
            
            all_xyz = []
            pcd_files = list(pcd_root.rglob("*.npy"))
            
            if pcd_files:
                for pcd_file in tqdm(pcd_files[:100], desc="Sampling PCD files"):  # Sample first 100 for speed
                    try:
                        pcd = np.load(pcd_file)
                        if pcd.ndim == 2 and pcd.shape[1] >= 3:
                            all_xyz.append(pcd[:, :3])
                    except Exception as e:
                        continue
                
                if all_xyz:
                    all_xyz = np.concatenate(all_xyz, axis=0)
                    pcd_stats = {
                        'xyz': {
                            'min': all_xyz.min(axis=0).tolist(),
                            'max': all_xyz.max(axis=0).tolist(),
                            'mean': all_xyz.mean(axis=0).tolist(),
                            'std': all_xyz.std(axis=0).tolist(),
                        }
                    }
                    
                    pcd_stats_path = output_dir / "pcd_stats.json"
                    with open(pcd_stats_path, 'w') as f:
                        json.dump(pcd_stats, f, indent=2)
                    print(f"✓ Saved PCD stats to: {pcd_stats_path}")
                    
                    print(f"\nPCD XYZ Statistics:")
                    print(f"  Min: {[f'{x:.4f}' for x in pcd_stats['xyz']['min']]}")
                    print(f"  Max: {[f'{x:.4f}' for x in pcd_stats['xyz']['max']]}")
            else:
                print(f"  No .npy files found in {pcd_root}")
    
    print(f"\n" + "="*80)
    print("DONE!")
    print("="*80)
    
    return action_stats, prop_stats


def main():
    parser = argparse.ArgumentParser(
        description='Compute statistics for BigYM Native Format training'
    )
    parser.add_argument(
        '--hdf5-path',
        type=str,
        required=True,
        help='Path to HDF5 file'
    )
    parser.add_argument(
        '--pcd-root',
        type=str,
        default=None,
        help='Path to PCD directory (optional, for PCD stats)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: same as HDF5 parent)'
    )
    
    args = parser.parse_args()
    compute_stats_bigym(args.hdf5_path, args.output_dir, args.pcd_root)


if __name__ == '__main__':
    main()
