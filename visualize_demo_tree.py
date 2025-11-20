"""
Create a beautiful ASCII tree visualization of HDF5 structure
"""
import h5py
import numpy as np
from pathlib import Path

def create_tree_visualization(hdf5_path: str, demo_id: str = "demo_1"):
    """
    Create beautiful tree structure visualization
    """
    
    with h5py.File(hdf5_path, 'r') as f:
        print("\n" + "═" * 100)
        print(f"📦 HDF5 FILE: {Path(hdf5_path).name}")
        print("═" * 100)
        
        if demo_id not in f['data']:
            print(f"Demo {demo_id} not found!")
            return
        
        demo = f['data'][demo_id]
        
        def format_shape(shape):
            if len(shape) == 0:
                return "scalar"
            return " × ".join(str(s) for s in shape)
        
        def format_size(nbytes):
            if nbytes < 1024:
                return f"{nbytes} B"
            elif nbytes < 1024**2:
                return f"{nbytes/1024:.1f} KB"
            elif nbytes < 1024**3:
                return f"{nbytes/(1024**2):.1f} MB"
            else:
                return f"{nbytes/(1024**3):.2f} GB"
        
        print(f"\n📁 data/")
        print(f"│")
        print(f"├── 📁 {demo_id}/")
        
        # Get all items
        items = list(demo.keys())
        
        # Process actions first
        if 'actions' in items:
            items.remove('actions')
            items.insert(0, 'actions')
        
        # Process groups
        obs_idx = [i for i, x in enumerate(items) if x == 'obs'][0] if 'obs' in items else -1
        observations_idx = [i for i, x in enumerate(items) if x == 'observations'][0] if 'observations' in items else -1
        
        for idx, key in enumerate(items):
            is_last = idx == len(items) - 1
            prefix = "└──" if is_last else "├──"
            continuation = "    " if is_last else "│   "
            
            item = demo[key]
            
            if isinstance(item, h5py.Dataset):
                # Dataset
                shape_str = format_shape(item.shape)
                size_str = format_size(item.nbytes)
                dtype_str = str(item.dtype)
                
                # Get data stats
                data = item[()]
                if np.issubdtype(item.dtype, np.number) and item.size > 0:
                    flat = data.flatten()
                    stats = f"[{flat.min():.3f}, {flat.max():.3f}]"
                else:
                    stats = ""
                
                print(f"│   {prefix} 📄 {key}")
                print(f"│   {continuation}   ├─ Shape: {shape_str}")
                print(f"│   {continuation}   ├─ Dtype: {dtype_str}")
                print(f"│   {continuation}   ├─ Size: {size_str}")
                if stats:
                    print(f"│   {continuation}   └─ Range: {stats}")
                else:
                    print(f"│   {continuation}   └─ Type: {dtype_str}")
                
            elif isinstance(item, h5py.Group):
                # Group
                print(f"│   {prefix} 📁 {key}/")
                
                subitems = list(item.keys())
                for sub_idx, subkey in enumerate(subitems):
                    is_sub_last = sub_idx == len(subitems) - 1
                    sub_prefix = "└──" if is_sub_last else "├──"
                    sub_continuation = "    " if is_sub_last else "│   "
                    
                    subitem = item[subkey]
                    
                    if isinstance(subitem, h5py.Dataset):
                        shape_str = format_shape(subitem.shape)
                        size_str = format_size(subitem.nbytes)
                        dtype_str = str(subitem.dtype)
                        
                        # Get data stats
                        data = subitem[()]
                        if np.issubdtype(subitem.dtype, np.number) and subitem.size > 0:
                            flat = data.flatten()
                            stats = f"[{flat.min():.2f}, {flat.max():.2f}]"
                        else:
                            stats = ""
                        
                        print(f"│   {continuation}{sub_prefix} 📄 {subkey}")
                        print(f"│   {continuation}{sub_continuation}   ├─ Shape: {shape_str}")
                        print(f"│   {continuation}{sub_continuation}   ├─ Dtype: {dtype_str}")
                        print(f"│   {continuation}{sub_continuation}   ├─ Size: {size_str}")
                        if stats:
                            print(f"│   {continuation}{sub_continuation}   └─ Range: {stats}")
                    
                    elif isinstance(subitem, h5py.Group):
                        # Nested group (e.g., point_cloud)
                        print(f"│   {continuation}{sub_prefix} 📁 {subkey}/")
                        
                        nested_items = list(subitem.keys())
                        for nest_idx, nestkey in enumerate(nested_items):
                            is_nest_last = nest_idx == len(nested_items) - 1
                            nest_prefix = "└──" if is_nest_last else "├──"
                            nest_continuation = "    " if is_nest_last else "│   "
                            
                            nestitem = subitem[nestkey]
                            if isinstance(nestitem, h5py.Dataset):
                                shape_str = format_shape(nestitem.shape)
                                size_str = format_size(nestitem.nbytes)
                                dtype_str = str(nestitem.dtype)
                                
                                data = nestitem[()]
                                if np.issubdtype(nestitem.dtype, np.number) and nestitem.size > 0:
                                    flat = data.flatten()
                                    stats = f"[{flat.min():.2f}, {flat.max():.2f}]"
                                else:
                                    stats = ""
                                
                                print(f"│   {continuation}{sub_continuation}{nest_prefix} 📄 {nestkey}")
                                print(f"│   {continuation}{sub_continuation}{nest_continuation}   ├─ Shape: {shape_str}")
                                print(f"│   {continuation}{sub_continuation}{nest_continuation}   ├─ Size: {size_str}")
                                if stats:
                                    print(f"│   {continuation}{sub_continuation}{nest_continuation}   └─ Range: {stats}")
        
        # Summary
        print("\n" + "─" * 100)
        print("📊 SUMMARY")
        print("─" * 100)
        
        if 'actions' in demo:
            actions = demo['actions'][()]
            print(f"\n🎯 Actions (GT Labels)")
            print(f"   Shape: {actions.shape}")
            print(f"   Structure (16-dim):")
            print(f"   ├─ [0:3]   Mobile base (x_delta, y_delta, rz_delta)")
            print(f"   ├─ [3:4]   Torso (z_delta)")
            print(f"   ├─ [4:9]   Left arm (5 joints)")
            print(f"   ├─ [9:10]  Left gripper")
            print(f"   ├─ [10:15] Right arm (5 joints)")
            print(f"   └─ [15:16] Right gripper")
            print(f"   Note: Deltas for mobile base & torso, absolutes for arms")
        
        if 'obs' in demo:
            obs = demo['obs']
            print(f"\n👁️  Observations")
            if 'proprioception' in obs:
                prop = obs['proprioception'][()]
                print(f"   ├─ Proprioception: {prop.shape} (robot joint states)")
            if 'proprioception_floating_base' in obs:
                fb = obs['proprioception_floating_base'][()]
                print(f"   ├─ Floating base: {fb.shape} (base orientation & position)")
            if 'proprioception_grippers' in obs:
                grippers = obs['proprioception_grippers'][()]
                print(f"   ├─ Grippers: {grippers.shape} (left, right gripper states)")
            if 'proprioception_floating_base_actions' in obs:
                fb_actions = obs['proprioception_floating_base_actions'][()]
                print(f"   └─ Floating base actions: {fb_actions.shape} (accumulated base actions)")
        
        if 'observations' in demo and 'point_cloud' in demo['observations']:
            pcd = demo['observations']['point_cloud']
            print(f"\n☁️  Point Clouds")
            for camera in pcd.keys():
                pcd_data = pcd[camera][()]
                print(f"   ├─ {camera}: {pcd_data.shape}")
                if len(pcd_data.shape) == 3 and pcd_data.shape[2] == 6:
                    print(f"   │  └─ Format: (frames, points, 6) = (frames, points, [X,Y,Z,R,G,B])")
        
        # Data flow
        print(f"\n🔄 Data Flow (Training)")
        print(f"   HDF5 'actions'")
        print(f"   └─→ PCDBRSDataset.__getitem__()")
        print(f"       ├─ Convert deltas → velocity/absolute")
        print(f"       ├─ Normalize to [-1, 1]")
        print(f"       └─ Split into: mobile_base(3) + torso(1) + arms(12)")
        print(f"           └─→ batch['action_chunks']")
        print(f"               └─→ DiffusionModule.training_step()")
        print(f"                   └─→ gt_actions (Ground Truth Labels)")
        print(f"                       └─→ Diffusion Loss Computation")
        
        print("\n" + "═" * 100)

def main():
    hdf5_path = "../data/demonstrations/0.9.0/SaucepanToHob_successful.hdf5"
    
    if not Path(hdf5_path).exists():
        print(f"File not found: {hdf5_path}")
        return
    
    create_tree_visualization(hdf5_path, demo_id="demo_1")

if __name__ == "__main__":
    main()
