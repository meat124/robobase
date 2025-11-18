"""
Compute proprioception statistics from HDF5 dataset for normalization
"""
import h5py
import numpy as np
import json
from pathlib import Path

def compute_prop_stats(hdf5_path: str, output_json: str = None):
    """
    Compute min/max statistics for proprioception from HDF5 dataset
    
    Args:
        hdf5_path: Path to HDF5 file
        output_json: Path to save JSON output (optional)
    
    Returns:
        dict: Statistics dictionary with min/max for each prop component
    """
    print(f"Loading HDF5: {hdf5_path}")
    f = h5py.File(hdf5_path, 'r')
    
    all_prop = []
    all_gripper = []
    demo_ids = list(f['data'].keys())
    
    print(f"Found {len(demo_ids)} demos")
    
    for demo_id in demo_ids:
        prop = f['data'][demo_id]['obs']['proprioception'][:]
        gripper = f['data'][demo_id]['obs']['proprioception_grippers'][:]
        all_prop.append(prop)
        all_gripper.append(gripper)
        print(f"  {demo_id}: {prop.shape[0]} frames")
    
    # Concatenate all proprioception
    all_prop = np.concatenate(all_prop, axis=0)
    all_gripper = np.concatenate(all_gripper, axis=0)
    print(f"\nTotal prop: {all_prop.shape}, gripper: {all_gripper.shape}")
    
    # ====================================================================
    # CRITICAL FIX: Use floating_base_actions instead of qvel
    # ====================================================================
    # Load floating base accumulated actions
    # floating_base_actions shape: (num_frames, 4) - X, Y, Z, RZ
    # We split this into mobile_base (X, Y, RZ) and torso (Z)
    all_floating_base = []
    for demo_id in demo_ids:
        floating_base = f['data'][demo_id]['obs']['proprioception_floating_base_actions'][:]
        all_floating_base.append(floating_base)
    all_floating_base = np.concatenate(all_floating_base, axis=0)
    print(f"Total floating_base_actions: {all_floating_base.shape}")
    
    # Extract mobile_base (X, Y, RZ) and torso (Z) from floating_base_actions
    mobile_base_xy = all_floating_base[:, 0:2]  # X, Y
    mobile_base_rz = all_floating_base[:, 3:4]  # RZ
    mobile_base_combined = np.concatenate([mobile_base_xy, mobile_base_rz], axis=1)  # (N, 3)
    torso_z = all_floating_base[:, 2:3]  # Z (pelvis_z)
    
    # Compute statistics for each component (matching dataset structure)
    stats = {
        "mobile_base_vel": {  # floating_base_actions X, Y, RZ (accumulated position)
            "min": mobile_base_combined.min(axis=0).tolist(),
            "max": mobile_base_combined.max(axis=0).tolist(),
            "mean": mobile_base_combined.mean(axis=0).tolist(),
            "std": mobile_base_combined.std(axis=0).tolist(),
        },
        "torso": {  # floating_base_actions Z (pelvis_z accumulated position)
            "min": float(torso_z.min()),
            "max": float(torso_z.max()),
            "mean": float(torso_z.mean()),
            "std": float(torso_z.std()),
        },
        "left_arm": {  # qpos[0:4, 12] - non-consecutive!
            "min": np.concatenate([all_prop[:, 0:4], all_prop[:, 12:13]], axis=1).min(axis=0).tolist(),
            "max": np.concatenate([all_prop[:, 0:4], all_prop[:, 12:13]], axis=1).max(axis=0).tolist(),
            "mean": np.concatenate([all_prop[:, 0:4], all_prop[:, 12:13]], axis=1).mean(axis=0).tolist(),
            "std": np.concatenate([all_prop[:, 0:4], all_prop[:, 12:13]], axis=1).std(axis=0).tolist(),
        },
        "left_gripper": {  # gripper[:, 0]
            "min": float(all_gripper[:, 0].min()),
            "max": float(all_gripper[:, 0].max()),
            "mean": float(all_gripper[:, 0].mean()),
            "std": float(all_gripper[:, 0].std()),
        },
        "right_arm": {  # qpos[13:17, 25] - non-consecutive!
            "min": np.concatenate([all_prop[:, 13:17], all_prop[:, 25:26]], axis=1).min(axis=0).tolist(),
            "max": np.concatenate([all_prop[:, 13:17], all_prop[:, 25:26]], axis=1).max(axis=0).tolist(),
            "mean": np.concatenate([all_prop[:, 13:17], all_prop[:, 25:26]], axis=1).mean(axis=0).tolist(),
            "std": np.concatenate([all_prop[:, 13:17], all_prop[:, 25:26]], axis=1).std(axis=0).tolist(),
        },
        "right_gripper": {  # gripper[:, 1]
            "min": float(all_gripper[:, 1].min()),
            "max": float(all_gripper[:, 1].max()),
            "mean": float(all_gripper[:, 1].mean()),
            "std": float(all_gripper[:, 1].std()),
        },
    }
    
    # Print statistics
    print("\n" + "="*80)
    print("PROPRIOCEPTION STATISTICS")
    print("="*80)
    
    print("\nMobile Base Velocity (qvel[30,31,33]):")
    for i, name in enumerate(['x_vel', 'y_vel', 'rz_vel']):
        print(f"  {name}: min={stats['mobile_base_vel']['min'][i]:8.5f}, "
              f"max={stats['mobile_base_vel']['max'][i]:8.5f}, "
              f"mean={stats['mobile_base_vel']['mean'][i]:8.5f}, "
              f"std={stats['mobile_base_vel']['std'][i]:8.5f}")
    
    print("\nTorso (qpos[2]):")
    print(f"  min={stats['torso']['min']:8.5f}, "
          f"max={stats['torso']['max']:8.5f}, "
          f"mean={stats['torso']['mean']:8.5f}, "
          f"std={stats['torso']['std']:8.5f}")
    
    print("\nLeft Arm (qpos[4:9]):")
    for i in range(5):
        print(f"  Joint {i}: min={stats['left_arm']['min'][i]:8.5f}, "
              f"max={stats['left_arm']['max'][i]:8.5f}, "
              f"mean={stats['left_arm']['mean'][i]:8.5f}, "
              f"std={stats['left_arm']['std'][i]:8.5f}")
    
    print("\nLeft Gripper:")
    print(f"  min={stats['left_gripper']['min']:8.5f}, "
          f"max={stats['left_gripper']['max']:8.5f}, "
          f"mean={stats['left_gripper']['mean']:8.5f}, "
          f"std={stats['left_gripper']['std']:8.5f}")
    
    print("\nRight Arm (qpos[17:22]):")
    for i in range(5):
        print(f"  Joint {i}: min={stats['right_arm']['min'][i]:8.5f}, "
              f"max={stats['right_arm']['max'][i]:8.5f}, "
              f"mean={stats['right_arm']['mean'][i]:8.5f}, "
              f"std={stats['right_arm']['std'][i]:8.5f}")
    
    print("\nRight Gripper:")
    print(f"  min={stats['right_gripper']['min']:8.5f}, "
          f"max={stats['right_gripper']['max']:8.5f}, "
          f"mean={stats['right_gripper']['mean']:8.5f}, "
          f"std={stats['right_gripper']['std']:8.5f}")
    
    # Save to JSON
    if output_json:
        output_path = Path(output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"\nSaved statistics to: {output_json}")
    
    f.close()
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute proprioception statistics")
    parser.add_argument("--input", type=str, 
                       default="../data/demonstrations/0.9.0/SaucepanToHob_successful.hdf5",
                       help="Input HDF5 file")
    parser.add_argument("--output", type=str, 
                       default="../data/demonstrations/0.9.0/prop_stats.json",
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    stats = compute_prop_stats(args.input, args.output)
