"""
Convert HDF5 dataset to Robobase-compatible format.

This script converts BigYM HDF5 demonstration files into a format
that can be loaded by Robobase's BiGymEnvFactory.

Usage:
    python convert_hdf5_to_robobase.py \
        --hdf5 ../data/demonstrations/0.9.0/SaucepanToHob_successful.hdf5 \
        --output ./converted_demos/saucepan/ \
        --task SaucepanToHob
"""

import h5py
import numpy as np
from pathlib import Path
import pickle
import argparse
from tqdm import tqdm
import sys

# Add BigYM to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'bigym'))

from demonstrations.demo import Demo, DemoStep
from demonstrations.utils import Metadata, ObservationMode


def extract_metadata_from_hdf5(hdf5_file, task_name):
    """Extract metadata from HDF5 file."""
    # Get any demo to extract info
    demo_ids = list(hdf5_file['data'].keys())
    if not demo_ids:
        raise ValueError("No demos found in HDF5 file")
    
    first_demo = hdf5_file['data'][demo_ids[0]]
    
    # Extract observation space info
    obs_keys = list(first_demo['obs'].keys())
    
    # Create metadata
    metadata = Metadata(
        task_name=task_name,
        observation_mode=ObservationMode.State,  # State mode for RGB observations
        floating_objects={},  # Will be populated later if needed
        demo_outcome=1,  # Successful demo
    )
    
    return metadata


def convert_hdf5_demo_to_bigym(demo_data, demo_id: str, metadata: Metadata):
    """
    Convert a single HDF5 demo to BiGym Demo format.
    
    Args:
        demo_data: HDF5 group containing demo data
        demo_id: Demo identifier (e.g., 'demo_1')
        metadata: Metadata object for the demo
        
    Returns:
        Demo object compatible with BiGym
    """
    # Extract data from HDF5
    actions = demo_data['actions'][:]  # Shape: (num_frames, action_dim)
    num_frames = len(actions)
    
    # Extract observations
    obs_group = demo_data['obs']
    obs_keys = list(obs_group.keys())
    
    # Build timesteps
    timesteps = []
    
    for i in range(num_frames):
        # Build observation dictionary
        observation = {}
        
        for key in obs_keys:
            obs_data = obs_group[key]
            
            if key == 'proprioception':
                # Proprioception is typically a flat array
                observation[key] = np.array(obs_data[i], dtype=np.float32)
                
            elif key.startswith('rgb_'):
                # RGB images
                # HDF5 shape: (num_frames, H, W, 3)
                # BiGym expects: (H, W, 3)
                img = obs_data[i]
                if img.dtype == np.uint8:
                    # Keep as uint8 for memory efficiency
                    observation[key] = img
                else:
                    # Convert to float32 if needed
                    observation[key] = np.array(img, dtype=np.float32)
                    
            elif key.startswith('depth_'):
                # Depth images
                depth = obs_data[i]
                observation[key] = np.array(depth, dtype=np.float32)
                
            else:
                # Other observation types
                observation[key] = np.array(obs_data[i], dtype=np.float32)
        
        # Get action (executed action that led to this observation)
        action = actions[i]
        
        # Determine reward (sparse reward at the end)
        # For successful demos, give reward 1.0 at the last step
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
        }
        
        # Create DemoStep
        demo_step = DemoStep(
            observation=observation,
            reward=reward,
            termination=termination,
            truncation=truncation,
            info=info,
            action=action,
        )
        
        timesteps.append(demo_step)
    
    # Create Demo object
    demo = Demo(
        metadata=metadata,
        timesteps=timesteps,
    )
    
    return demo


def convert_hdf5_to_demos(
    hdf5_path: str,
    output_dir: str,
    task_name: str,
    max_demos: int = None,
):
    """
    Convert HDF5 dataset to BiGym demo format.
    
    Args:
        hdf5_path: Path to HDF5 file
        output_dir: Output directory for demos
        task_name: Task name (e.g., 'SaucepanToHob')
        max_demos: Maximum number of demos to convert (None for all)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting HDF5 dataset: {hdf5_path}")
    print(f"Output directory: {output_dir}")
    print(f"Task name: {task_name}")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Extract metadata
        metadata = extract_metadata_from_hdf5(f, task_name)
        
        # Get all demo IDs
        demo_ids = sorted(list(f['data'].keys()), key=lambda x: int(x.split('_')[1]))
        
        if max_demos is not None:
            demo_ids = demo_ids[:max_demos]
        
        print(f"Found {len(demo_ids)} demos to convert")
        
        # Convert each demo
        for demo_id in tqdm(demo_ids, desc="Converting demos"):
            demo_data = f['data'][demo_id]
            
            # Convert to BiGym format
            demo = convert_hdf5_demo_to_bigym(demo_data, demo_id, metadata)
            
            # Save as pickle (BiGym format)
            demo_file = output_dir / f'{demo_id}.pkl'
            with open(demo_file, 'wb') as f_out:
                pickle.dump(demo, f_out)
            
            print(f'  Saved {demo_id}: {len(demo.timesteps)} steps')
    
    print(f'\n✓ Converted {len(demo_ids)} demos to {output_dir}')
    
    # Print summary
    print("\nSummary:")
    print(f"  Total demos: {len(demo_ids)}")
    print(f"  Output format: BiGym Demo (pickle)")
    print(f"  Task: {task_name}")
    
    # Print first demo info
    first_demo_path = output_dir / f'{demo_ids[0]}.pkl'
    with open(first_demo_path, 'rb') as f:
        first_demo = pickle.load(f)
    
    print(f"\nFirst demo ({demo_ids[0]}) info:")
    print(f"  Number of steps: {len(first_demo.timesteps)}")
    print(f"  Observation keys: {list(first_demo.timesteps[0].observation.keys())}")
    print(f"  Action shape: {first_demo.timesteps[0].executed_action.shape}")
    
    # Check observation shapes
    first_obs = first_demo.timesteps[0].observation
    print(f"\nObservation shapes:")
    for key, value in first_obs.items():
        print(f"    {key}: {value.shape} ({value.dtype})")


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
    
    print(f"\nVerifying {num_samples} demos from {demo_dir}")
    
    for demo_file in demo_files[:num_samples]:
        try:
            with open(demo_file, 'rb') as f:
                demo = pickle.load(f)
            
            print(f"\n✓ {demo_file.name}:")
            print(f"    Steps: {len(demo.timesteps)}")
            print(f"    Metadata: {demo._metadata.task_name}")
            print(f"    First action: {demo.timesteps[0].executed_action[:5]}...")
            print(f"    Last reward: {demo.timesteps[-1].reward}")
            print(f"    Termination: {demo.timesteps[-1].termination}")
            
        except Exception as e:
            print(f"✗ Failed to load {demo_file.name}: {e}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert HDF5 dataset to Robobase format'
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
        '--task',
        type=str,
        default='SaucepanToHob',
        help='Task name (default: SaucepanToHob)'
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
    convert_hdf5_to_demos(
        hdf5_path=args.hdf5,
        output_dir=args.output,
        task_name=args.task,
        max_demos=args.max_demos,
    )
    
    # Verify if requested
    if args.verify:
        verify_converted_demos(args.output)


if __name__ == '__main__':
    main()
