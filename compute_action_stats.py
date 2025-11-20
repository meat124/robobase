"""
Compute action statistics from HDF5 dataset for normalization
"""
import h5py
import numpy as np
import json
from pathlib import Path

def compute_action_stats(hdf5_path: str, output_json: str = None):
    """
    Compute min/max statistics for actions from HDF5 dataset
    
    Args:
        hdf5_path: Path to HDF5 file
        output_json: Path to save JSON output (optional)
    
    Returns:
        dict: Statistics dictionary with min/max for each action component
    """
    print(f"Loading HDF5: {hdf5_path}")
    f = h5py.File(hdf5_path, 'r')
    
    # Control frequency for velocity conversion
    dt = 0.02  # 50Hz = 1/50 = 0.02s
    
    all_mobile_base = []
    all_torso_absolute = []
    all_arms = []
    demo_ids = list(f['data'].keys())
    
    print(f"Found {len(demo_ids)} demos")
    
    for demo_id in demo_ids:
        actions = f['data'][demo_id]['actions'][:]
        
        # Convert mobile base deltas to velocity
        mobile_base_velocity = actions[:, [0, 1, 3]].copy()  # X, Y, RZ
        mobile_base_velocity /= dt
        
        # Convert torso deltas to absolute positions
        # Get initial torso position from floating_base_actions
        floating_base = f['data'][demo_id]['obs']['proprioception_floating_base_actions'][:]
        initial_torso_pos = floating_base[0, 2]  # Initial Z position
        torso_deltas = actions[:, 2]
        torso_absolute = initial_torso_pos + np.cumsum(torso_deltas)  # Accumulate deltas
        
        # Arms are already absolute positions
        arms = actions[:, 4:16]
        
        all_mobile_base.append(mobile_base_velocity)
        all_torso_absolute.append(torso_absolute)
        all_arms.append(arms)
        
        print(f"  {demo_id}: {actions.shape[0]} frames")
    
    # Concatenate all actions
    all_mobile_base = np.concatenate(all_mobile_base, axis=0)
    all_torso_absolute = np.concatenate(all_torso_absolute, axis=0)
    all_arms = np.concatenate(all_arms, axis=0)
    
    print(f"\nTotal frames: {all_mobile_base.shape[0]}")
    
    # Compute statistics
    stats = {
        "mobile_base": {
            "min": all_mobile_base.min(axis=0).tolist(),
            "max": all_mobile_base.max(axis=0).tolist(),
            "mean": all_mobile_base.mean(axis=0).tolist(),
            "std": all_mobile_base.std(axis=0).tolist(),
        },
        "torso": {
            "min": float(all_torso_absolute.min()),
            "max": float(all_torso_absolute.max()),
            "mean": float(all_torso_absolute.mean()),
            "std": float(all_torso_absolute.std()),
        },
        "arms": {
            "min": all_arms.min(axis=0).tolist(),
            "max": all_arms.max(axis=0).tolist(),
            "mean": all_arms.mean(axis=0).tolist(),
            "std": all_arms.std(axis=0).tolist(),
        },
    }
    
    # Print statistics
    print("\n" + "="*80)
    print("ACTION STATISTICS")
    print("="*80)
    
    print("\nMobile Base (X, Y, RZ) [VELOCITY m/s or rad/s]:")
    for i, (name) in enumerate(['x', 'y', 'rz']):
        print(f"  {name}: min={stats['mobile_base']['min'][i]:8.5f}, "
              f"max={stats['mobile_base']['max'][i]:8.5f}, "
              f"mean={stats['mobile_base']['mean'][i]:8.5f}, "
              f"std={stats['mobile_base']['std'][i]:8.5f}")
    
    print("\nTorso (Z - pelvis_z) [ABSOLUTE POSITION m]:")
    print(f"  min={stats['torso']['min']:8.5f}, "
          f"max={stats['torso']['max']:8.5f}, "
          f"mean={stats['torso']['mean']:8.5f}, "
          f"std={stats['torso']['std']:8.5f}")
    
    print("\nArms (4-15):")
    for i in range(12):
        print(f"  Dim {i:2d}: min={stats['arms']['min'][i]:8.5f}, "
              f"max={stats['arms']['max'][i]:8.5f}, "
              f"mean={stats['arms']['mean'][i]:8.5f}, "
              f"std={stats['arms']['std'][i]:8.5f}")
    
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
    
    parser = argparse.ArgumentParser(description="Compute action statistics")
    parser.add_argument("--input", type=str, 
                       default="../data/demonstrations/0.9.0/SaucepanToHob_successful.hdf5",
                       help="Input HDF5 file")
    parser.add_argument("--output", type=str, 
                       default="../data/demonstrations/0.9.0/action_stats.json",
                       help="Output JSON file")
    
    args = parser.parse_args()
    
    stats = compute_action_stats(args.input, args.output)
