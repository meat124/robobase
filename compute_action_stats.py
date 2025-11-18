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
    
    all_actions = []
    demo_ids = list(f['data'].keys())
    
    print(f"Found {len(demo_ids)} demos")
    
    for demo_id in demo_ids:
        actions = f['data'][demo_id]['actions'][:]
        
        # NOTE: Actions are DELTA POSITION (not velocity)
        # BigYM uses JointPositionActionMode with delta control
        # No conversion needed - use delta directly for statistics
        # Typical delta ranges: mobile_base ~0.01m, torso ~0.01m, arms ~0.5rad
        
        all_actions.append(actions)  # Use delta directly
        print(f"  {demo_id}: {actions.shape[0]} frames")
    
    # Concatenate all actions
    all_actions = np.concatenate(all_actions, axis=0)
    print(f"\nTotal actions: {all_actions.shape}")
    
    # Compute statistics
    # Mobile base: [X, Y] (indices 0-1) + [RZ] (index 3)
    # Torso: [Z] (index 2)
    # Arms: [4:16]
    mobile_base_xy = all_actions[:, 0:2]
    mobile_base_rz = all_actions[:, 3:4]
    mobile_base = np.concatenate([mobile_base_xy, mobile_base_rz], axis=1)  # Shape: (N, 3)
    
    stats = {
        "mobile_base": {
            "min": mobile_base.min(axis=0).tolist(),
            "max": mobile_base.max(axis=0).tolist(),
            "mean": mobile_base.mean(axis=0).tolist(),
            "std": mobile_base.std(axis=0).tolist(),
        },
        "torso": {
            "min": float(all_actions[:, 2].min()),
            "max": float(all_actions[:, 2].max()),
            "mean": float(all_actions[:, 2].mean()),
            "std": float(all_actions[:, 2].std()),
        },
        "arms": {
            "min": all_actions[:, 4:16].min(axis=0).tolist(),
            "max": all_actions[:, 4:16].max(axis=0).tolist(),
            "mean": all_actions[:, 4:16].mean(axis=0).tolist(),
            "std": all_actions[:, 4:16].std(axis=0).tolist(),
        },
    }
    
    # Print statistics
    print("\n" + "="*80)
    print("ACTION STATISTICS (DELTA POSITION)")
    print("="*80)
    
    print("\nMobile Base (X, Y, RZ) [DELTA POSITION m or rad]:")
    for i, (name) in enumerate(['x', 'y', 'rz']):
        print(f"  {name}: min={stats['mobile_base']['min'][i]:8.5f}, "
              f"max={stats['mobile_base']['max'][i]:8.5f}, "
              f"mean={stats['mobile_base']['mean'][i]:8.5f}, "
              f"std={stats['mobile_base']['std'][i]:8.5f}")
    
    print("\nTorso (Z - pelvis_z) [DELTA POSITION m]:")
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
