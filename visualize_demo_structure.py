"""
Complete visualization of HDF5 demo_data structure
"""
import h5py
import numpy as np
from pathlib import Path
import sys

def print_attrs(name, obj):
    """Print attributes of an HDF5 object"""
    if obj.attrs:
        print(f"      Attributes:")
        for key, val in obj.attrs.items():
            print(f"        - {key}: {val}")

def visualize_hdf5_structure(hdf5_path: str, demo_id: str = None, max_demos: int = 3):
    """
    Completely visualize HDF5 structure with all details
    
    Args:
        hdf5_path: Path to HDF5 file
        demo_id: Specific demo to inspect (None = show first few)
        max_demos: Maximum number of demos to show if demo_id is None
    """
    print("=" * 80)
    print(f"HDF5 FILE STRUCTURE VISUALIZATION")
    print(f"File: {hdf5_path}")
    print("=" * 80)
    
    with h5py.File(hdf5_path, 'r') as f:
        # Show root level structure
        print("\n[ROOT LEVEL]")
        print(f"Keys: {list(f.keys())}")
        
        if 'data' not in f:
            print("ERROR: 'data' group not found!")
            return
        
        data_group = f['data']
        all_demos = list(data_group.keys())
        print(f"\nTotal demos in file: {len(all_demos)}")
        print(f"Demo IDs: {all_demos[:10]}{'...' if len(all_demos) > 10 else ''}")
        
        # Determine which demos to inspect
        if demo_id:
            if demo_id not in all_demos:
                print(f"\nERROR: {demo_id} not found in file!")
                print(f"Available demos: {all_demos}")
                return
            demos_to_inspect = [demo_id]
        else:
            demos_to_inspect = all_demos[:max_demos]
        
        # Inspect each demo
        for idx, demo in enumerate(demos_to_inspect):
            print("\n" + "=" * 80)
            print(f"DEMO: {demo} ({idx+1}/{len(demos_to_inspect)})")
            print("=" * 80)
            
            demo_group = data_group[demo]
            
            def print_structure(group, indent=0):
                """Recursively print HDF5 structure"""
                prefix = "  " * indent
                
                for key in group.keys():
                    item = group[key]
                    
                    if isinstance(item, h5py.Group):
                        print(f"{prefix}📁 {key}/")
                        print_attrs(key, item)
                        print_structure(item, indent + 1)
                    
                    elif isinstance(item, h5py.Dataset):
                        data = item[()]
                        shape = item.shape
                        dtype = item.dtype
                        
                        print(f"{prefix}📄 {key}")
                        print(f"{prefix}   Shape: {shape}")
                        print(f"{prefix}   Dtype: {dtype}")
                        print(f"{prefix}   Size: {item.size:,} elements")
                        print(f"{prefix}   Memory: {item.nbytes / 1024 / 1024:.2f} MB")
                        
                        # Print attributes
                        print_attrs(key, item)
                        
                        # Show data sample
                        if shape == ():
                            # Scalar
                            print(f"{prefix}   Value: {data}")
                        elif len(shape) == 1 and shape[0] <= 10:
                            # Small 1D array
                            print(f"{prefix}   Data: {data}")
                        elif len(shape) >= 1:
                            # Multi-dimensional array
                            print(f"{prefix}   First element shape: {data[0].shape if len(data) > 0 else 'N/A'}")
                            if len(shape) == 1:
                                print(f"{prefix}   Sample [0:3]: {data[:3]}")
                            elif len(shape) == 2:
                                print(f"{prefix}   Sample [0]: {data[0][:5]}{'...' if data.shape[1] > 5 else ''}")
                            elif len(shape) == 3:
                                print(f"{prefix}   Sample [0,0]: {data[0,0][:5] if len(data[0,0]) > 0 else 'empty'}")
                            
                            # Statistics for numeric data
                            if np.issubdtype(dtype, np.number) and item.size > 0:
                                flat_data = data.flatten()
                                print(f"{prefix}   Stats:")
                                print(f"{prefix}     - Min: {np.min(flat_data):.6f}")
                                print(f"{prefix}     - Max: {np.max(flat_data):.6f}")
                                print(f"{prefix}     - Mean: {np.mean(flat_data):.6f}")
                                print(f"{prefix}     - Std: {np.std(flat_data):.6f}")
                        
                        print()
            
            print_structure(demo_group)
            
            # Special analysis for this demo
            print(f"\n{'─' * 80}")
            print(f"SPECIAL ANALYSIS FOR {demo}")
            print(f"{'─' * 80}")
            
            # Check if actions exist
            if 'actions' in demo_group:
                actions = demo_group['actions'][()]
                print(f"\n✓ Actions found: {actions.shape}")
                print(f"  Action dimensions (16):")
                print(f"    [0:3]   mobile_base (x_delta, y_delta, rz_delta): {actions[0, :3]}")
                print(f"    [3:4]   torso (z_delta): {actions[0, 3]}")
                print(f"    [4:9]   left_arm (5 joints): {actions[0, 4:9]}")
                print(f"    [9:10]  left_gripper: {actions[0, 9]}")
                print(f"    [10:15] right_arm (5 joints): {actions[0, 10:15]}")
                print(f"    [15:16] right_gripper: {actions[0, 15]}")
                
                # Check if deltas or absolutes
                print(f"\n  Checking if deltas or absolutes:")
                print(f"    Mobile base range: [{actions[:, :3].min():.4f}, {actions[:, :3].max():.4f}]")
                print(f"    Torso range: [{actions[:, 3].min():.4f}, {actions[:, 3].max():.4f}]")
                print(f"    Arms range: [{actions[:, 4:16].min():.4f}, {actions[:, 4:16].max():.4f}]")
            
            # Check obs structure
            if 'obs' in demo_group:
                obs_group = demo_group['obs']
                print(f"\n✓ Observations found")
                
                if 'proprioception' in obs_group:
                    prop = obs_group['proprioception'][()]
                    print(f"  Proprioception: {prop.shape}")
                    print(f"    First frame: {prop[0, :10]}...")
                
                if 'proprioception_floating_base' in obs_group:
                    fb = obs_group['proprioception_floating_base'][()]
                    print(f"  Floating base: {fb.shape}")
                    print(f"    First frame: {fb[0]}")
                
                if 'proprioception_grippers' in obs_group:
                    grippers = obs_group['proprioception_grippers'][()]
                    print(f"  Grippers: {grippers.shape}")
                    print(f"    First frame: {grippers[0]}")
                
                if 'proprioception_floating_base_actions' in obs_group:
                    fb_actions = obs_group['proprioception_floating_base_actions'][()]
                    print(f"  Floating base actions (accumulated): {fb_actions.shape}")
                    print(f"    First frame: {fb_actions[0]}")
                    print(f"    Last frame: {fb_actions[-1]}")
            
            # Check observations/point_cloud
            if 'observations' in demo_group:
                obs_group = demo_group['observations']
                if 'point_cloud' in obs_group:
                    pcd_group = obs_group['point_cloud']
                    print(f"\n✓ Point clouds found")
                    for camera in pcd_group.keys():
                        pcd_data = pcd_group[camera]
                        print(f"  Camera '{camera}': {pcd_data.shape}")
                        if len(pcd_data.shape) == 3:
                            print(f"    Frame 0: {pcd_data[0].shape} points")
                            if pcd_data[0].shape[1] >= 6:
                                print(f"    XYZ range: [{pcd_data[0, :, :3].min():.3f}, {pcd_data[0, :, :3].max():.3f}]")
                                print(f"    RGB range: [{pcd_data[0, :, 3:6].min():.3f}, {pcd_data[0, :, 3:6].max():.3f}]")
            
            # Timing analysis
            if 'actions' in demo_group:
                num_frames = demo_group['actions'].shape[0]
                dt = 0.02  # 50Hz
                duration = num_frames * dt
                print(f"\n✓ Timing:")
                print(f"  Total frames: {num_frames}")
                print(f"  Duration: {duration:.2f} seconds")
                print(f"  Frequency: {1/dt:.0f} Hz")

def main():
    # Default path
    hdf5_path = Path("../data/demonstrations/0.9.0/SaucepanToHob_successful.hdf5")
    
    # Check if file exists
    if not hdf5_path.exists():
        print(f"ERROR: File not found: {hdf5_path}")
        print("Please provide correct path as argument")
        return
    
    # Allow command line argument
    if len(sys.argv) > 1:
        hdf5_path = Path(sys.argv[1])
    
    demo_id = sys.argv[2] if len(sys.argv) > 2 else None
    max_demos = int(sys.argv[3]) if len(sys.argv) > 3 else 2
    
    visualize_hdf5_structure(str(hdf5_path), demo_id, max_demos)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
