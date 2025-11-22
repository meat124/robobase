"""
Convert HDF5 dataset to Robobase-compatible format (simplified version).

This script converts BigYM HDF5 demonstration files into a simple pickle format
that can be loaded by Robobase.

Features:
- Removes depth images (not used in training)
- Resizes RGB images to 84×84 (from 224×224) to save memory
- Keeps all proprioception data

Usage:
    python convert_hdf5_simple.py \
        --hdf5 ../data/demonstrations/0.9.0/SaucepanToHob_successful.hdf5 \
        --output ./converted_demos/saucepan/ \
        --max-demos 5
"""

import h5py
import numpy as np
from pathlib import Path
import pickle
import argparse
from tqdm import tqdm
from PIL import Image


def convert_hdf5_demo_simple(demo_data, demo_id: str):
    """
    Convert a single HDF5 demo to simple timestep format.
    
    Returns list of (observation, reward, termination, truncation, info) tuples
    Plus initial observation tuple: (observation, info)
    """
    # Extract data from HDF5
    actions = demo_data['actions'][:]  # Shape: (num_frames, action_dim)
    num_frames = len(actions)
    
    # Extract observations
    obs_group = demo_data['obs']
    
    # Build timesteps
    timesteps = []
    
    for i in range(num_frames):
        # Build observation dictionary
        observation = {}
        
        # RGB images: Resize from 224×224 to 84×84 and convert to (C, H, W) format
        for cam in ['rgb_head', 'rgb_left_wrist', 'rgb_right_wrist']:
            if cam in obs_group:
                img = obs_group[cam][i]  # Shape: (3, 224, 224) or (224, 224, 3)
                
                # Convert to channel-last format (H, W, C) for PIL
                if img.shape[0] == 3:  # (C, H, W) -> (H, W, C)
                    img = np.transpose(img, (1, 2, 0))
                
                # Convert to uint8 if needed for PIL
                if img.dtype != np.uint8:
                    img = (img * 255).astype(np.uint8)
                
                # Resize using PIL (high quality)
                img_pil = Image.fromarray(img)
                img_resized = img_pil.resize((84, 84), Image.BILINEAR)
                img_array = np.array(img_resized)
                
                # Convert back to float32 [0, 1] and channel-first (C, H, W)
                img_normalized = img_array.astype(np.float32) / 255.0
                img_chw = np.transpose(img_normalized, (2, 0, 1))  # (H, W, C) -> (C, H, W)
                
                observation[cam] = img_chw
        
        # SKIP depth images - not used in training
        # This saves ~0.57 MB per timestep
        
        # Proprioception fields - keep individual fields for ConcatDim wrapper
        # NOTE: BiGym environment observation space with torso enabled:
        #   - proprioception: (60,) - includes torso
        #   - proprioception_floating_base: (3,) - X, Y, RZ (no Z)
        #   - proprioception_floating_base_actions: (3,)
        if 'proprioception' in obs_group:
            observation['proprioception'] = obs_group['proprioception'][i].astype(np.float32)
        
        if 'proprioception_floating_base' in obs_group:
            # Use only X, Y, RZ (indices 0, 1, 3) - skip Z/torso (index 2)
            fb = obs_group['proprioception_floating_base'][i]
            observation['proprioception_floating_base'] = np.array([fb[0], fb[1], fb[3]], dtype=np.float32)
        
        if 'proprioception_grippers' in obs_group:
            observation['proprioception_grippers'] = obs_group['proprioception_grippers'][i].astype(np.float32)
        
        # Proprioception floating base actions (required by replay buffer, kept separate)
        if 'proprioception_floating_base_actions' in obs_group:
            # Also use only X, Y, RZ (indices 0, 1, 3)
            fba = obs_group['proprioception_floating_base_actions'][i]
            observation['proprioception_floating_base_actions'] = np.array([fba[0], fba[1], fba[3]], dtype=np.float32)
        
        # NOTE: Do NOT create low_dim_state here! 
        # The ConcatDim wrapper will create it from individual proprioception fields.
        
        # Get action (executed action that led to this observation)
        action = actions[i].astype(np.float32)
        
        # Determine reward (sparse reward at the end)
        if i == num_frames - 1:
            reward = 1.0
            termination = True
            truncation = False
        else:
            reward = 0.0
            termination = False
            truncation = False
        
        # Build info dictionary
        info = {
            'demo': 1,  # Successful demo
            'demo_id': demo_id,
            'step': i,
            'demo_action': action,  # Store action in info for stats computation
        }
        
        # First timestep: only (observation, info)
        if i == 0:
            timesteps.append((observation, info))
        else:
            # Subsequent timesteps: (observation, reward, term, trunc, info)
            timesteps.append((observation, reward, termination, truncation, info))
    
    return timesteps


def convert_hdf5_to_demos(
    hdf5_path: str,
    output_dir: str,
    max_demos: int = None,
):
    """
    Convert HDF5 dataset to simple demo format.
    
    Args:
        hdf5_path: Path to HDF5 file
        output_dir: Output directory for demos
        max_demos: Maximum number of demos to convert (None for all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting HDF5 dataset: {hdf5_path}")
    print(f"Output directory: {output_dir}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Get all demo IDs
        demo_ids = sorted(list(f['data'].keys()), key=lambda x: int(x.split('_')[1]))
        
        if max_demos is not None:
            demo_ids = demo_ids[:max_demos]
        
        print(f"Found {len(demo_ids)} demos to convert")
        
        # Convert each demo
        for demo_id in tqdm(demo_ids, desc="Converting demos"):
            demo_data = f['data'][demo_id]
            
            # Convert to simple format
            timesteps = convert_hdf5_demo_simple(demo_data, demo_id)
            
            # Save as pickle
            demo_file = output_dir / f'{demo_id}.pkl'
            with open(demo_file, 'wb') as f_out:
                pickle.dump(timesteps, f_out)
            
            print(f'  Saved {demo_id}: {len(timesteps)} steps')
    
    print(f'\n✓ Converted {len(demo_ids)} demos to {output_dir}')
    
    return demo_ids


def verify_converted_demos(demo_dir: str, num_samples: int = 3):
    """
    Verify converted demos can be loaded correctly.
    
    Args:
        demo_dir: Directory containing converted demos
        num_samples: Number of demos to verify
    """
    demo_dir = Path(demo_dir)
    demo_files = sorted(demo_dir.glob('*.pkl'))
    
    if not demo_files:
        print(f"No demo files found in {demo_dir}")
        return
    
    print(f"\nVerifying {min(num_samples, len(demo_files))} demos from {demo_dir}")
    
    for demo_file in demo_files[:num_samples]:
        try:
            with open(demo_file, 'rb') as f:
                timesteps = pickle.load(f)
            
            # First timestep
            first_obs, first_info = timesteps[0]
            
            # Last timestep
            last_obs, last_reward, last_term, last_trunc, last_info = timesteps[-1]
            
            print(f"\n✓ {demo_file.name}:")
            print(f"    Steps: {len(timesteps)}")
            print(f"    Observation keys: {list(first_obs.keys())}")
            print(f"    First action: {first_info['demo_action'][:5]}...")
            print(f"    Last reward: {last_reward}")
            print(f"    Termination: {last_term}")
            
            # Check shapes
            print(f"    Observation shapes:")
            for key, value in first_obs.items():
                print(f"      {key}: {value.shape} ({value.dtype})")
            
        except Exception as e:
            print(f"✗ Failed to load {demo_file.name}: {e}")
            import traceback
            traceback.print_exc()


def print_summary(demo_ids, output_dir):
    """Print conversion summary."""
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Total demos converted: {len(demo_ids)}")
    print(f"Output directory: {output_dir}")
    print(f"Demo format: Timestep tuples (pickle)")
    print("\nDemo IDs:")
    for demo_id in demo_ids:
        print(f"  - {demo_id}")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Convert HDF5 dataset to Robobase format (simplified)'
    )
    parser.add_argument(
        '--hdf5',
        type=str,
        required=True,
        help='Path to HDF5 file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output directory for converted demos'
    )
    parser.add_argument(
        '--max-demos',
        type=int,
        default=None,
        help='Maximum number of demos to convert (default: all)'
    )
    parser.add_argument(
        '--verify',
        action='store_true',
        help='Verify converted demos after conversion'
    )
    
    args = parser.parse_args()
    
    # Convert HDF5 to demos
    demo_ids = convert_hdf5_to_demos(
        hdf5_path=args.hdf5,
        output_dir=args.output,
        max_demos=args.max_demos,
    )
    
    # Print summary
    print_summary(demo_ids, args.output)
    
    # Verify if requested
    if args.verify:
        verify_converted_demos(args.output)


if __name__ == '__main__':
    main()
