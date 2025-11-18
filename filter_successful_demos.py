#!/usr/bin/env python3
"""
Filter HDF5 dataset to include only successful episodes (those with PCD data)
"""

import h5py
import numpy as np
from pathlib import Path
import shutil


def get_successful_demos(pcd_root: str):
    """Get list of demo IDs that have PCD data (successful episodes)"""
    pcd_path = Path(pcd_root)
    demo_dirs = sorted([d.name for d in pcd_path.iterdir() if d.is_dir() and d.name.startswith('demo_')])
    print(f"Found {len(demo_dirs)} successful demos in {pcd_root}")
    print(f"Successful demos: {', '.join(demo_dirs)}")
    return demo_dirs


def filter_hdf5_by_demos(input_hdf5: str, output_hdf5: str, successful_demos: list):
    """
    Create new HDF5 file containing only successful demos
    
    Args:
        input_hdf5: Path to original HDF5 file
        output_hdf5: Path to output HDF5 file
        successful_demos: List of demo IDs to keep
    """
    
    print(f"\nFiltering HDF5: {input_hdf5}")
    print(f"Output: {output_hdf5}")
    
    # Open source file
    with h5py.File(input_hdf5, 'r') as src_f:
        all_demos = sorted(src_f['data'].keys())
        print(f"\nTotal demos in source: {len(all_demos)}")
        
        # Filter demos
        demos_to_keep = [d for d in all_demos if d in successful_demos]
        demos_to_remove = [d for d in all_demos if d not in successful_demos]
        
        print(f"Demos to keep: {len(demos_to_keep)}")
        print(f"Demos to remove: {len(demos_to_remove)}")
        
        if demos_to_remove:
            print(f"\nRemoving failed demos: {', '.join(demos_to_remove)}")
        
        # Create output file
        with h5py.File(output_hdf5, 'w') as dst_f:
            # Copy metadata group if exists
            if 'metadata' in src_f:
                src_f.copy('metadata', dst_f)
            
            # Create data group
            dst_data = dst_f.create_group('data')
            
            # Copy only successful demos
            total_frames = 0
            for demo_id in demos_to_keep:
                print(f"  Copying {demo_id}...", end=' ')
                src_demo = src_f['data'][demo_id]
                
                # Copy entire demo group
                src_f['data'].copy(demo_id, dst_data)
                
                # Count frames
                frames = len(src_demo['actions'])
                total_frames += frames
                print(f"{frames} frames")
            
            print(f"\n✅ Successfully created filtered HDF5")
            print(f"   Total demos: {len(demos_to_keep)}")
            print(f"   Total frames: {total_frames}")


def verify_filtered_hdf5(hdf5_path: str, expected_demos: list):
    """Verify the filtered HDF5 file"""
    print(f"\n{'='*80}")
    print("VERIFICATION")
    print('='*80)
    
    with h5py.File(hdf5_path, 'r') as f:
        demos = sorted(f['data'].keys())
        print(f"\nFiltered HDF5: {hdf5_path}")
        print(f"Total demos: {len(demos)}")
        print(f"Demos: {', '.join(demos)}")
        
        # Check if all expected demos are present
        missing = set(expected_demos) - set(demos)
        extra = set(demos) - set(expected_demos)
        
        if missing:
            print(f"\n⚠️  Missing demos: {', '.join(missing)}")
        if extra:
            print(f"\n⚠️  Extra demos: {', '.join(extra)}")
        
        if not missing and not extra:
            print("\n✅ All expected demos are present!")
        
        # Print frame counts
        print("\nFrame counts:")
        total_frames = 0
        for demo_id in demos:
            frames = len(f['data'][demo_id]['actions'])
            total_frames += frames
            print(f"  {demo_id}: {frames} frames")
        
        print(f"\nTotal frames: {total_frames}")
        print('='*80)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Filter HDF5 to include only successful demos")
    parser.add_argument("--input", type=str, 
                       default="../data/demonstrations/0.9.0/SaucepanToHob.hdf5",
                       help="Input HDF5 file")
    parser.add_argument("--output", type=str, 
                       default="../data/demonstrations/0.9.0/SaucepanToHob_successful.hdf5",
                       help="Output HDF5 file")
    parser.add_argument("--pcd-root", type=str,
                       default="../bigym/pcd_output_filtered",
                       help="PCD root directory to determine successful demos")
    parser.add_argument("--verify", action="store_true",
                       help="Verify the filtered HDF5 after creation")
    
    args = parser.parse_args()
    
    # Get successful demos from PCD directory
    successful_demos = get_successful_demos(args.pcd_root)
    
    # Filter HDF5
    filter_hdf5_by_demos(args.input, args.output, successful_demos)
    
    # Verify if requested
    if args.verify:
        verify_filtered_hdf5(args.output, successful_demos)
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. Recompute statistics with filtered HDF5:")
    print(f"   python compute_prop_stats.py --input {args.output}")
    print(f"   python compute_action_stats.py --input {args.output}")
    print("\n2. Update dataset configuration to use filtered HDF5")
    print("="*80)
