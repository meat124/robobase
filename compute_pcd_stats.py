"""
PCD 파일의 XYZ 좌표 통계(min/max)를 계산하여 정규화에 사용

Usage:
    python compute_pcd_stats.py \
        --pcd-root ../bigym/pcd_output_filtered \
        --cameras head left_wrist right_wrist \
        --output ../data/demonstrations/0.9.0/pcd_stats.json \
        --sample-rate 0.1
"""

import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from tqdm import tqdm


def compute_pcd_xyz_stats(pcd_root: Path, cameras: list[str], sample_rate: float = 0.1):
    """
    PCD 파일들에서 XYZ 좌표의 min/max를 계산
    
    Args:
        pcd_root: PCD 루트 디렉토리
        cameras: 카메라 목록
        sample_rate: 샘플링 비율 (0.1 = 10% 파일만 사용)
    """
    
    print(f"Computing PCD XYZ statistics from: {pcd_root}")
    print(f"Cameras: {cameras}")
    print(f"Sample rate: {sample_rate * 100:.1f}%")
    print()
    
    all_xyz_min = []
    all_xyz_max = []
    
    demo_dirs = sorted([d for d in pcd_root.iterdir() if d.is_dir()])
    print(f"Found {len(demo_dirs)} demos")
    
    total_frames_processed = 0
    
    for demo_dir in tqdm(demo_dirs, desc="Processing demos"):
        for camera in cameras:
            camera_dir = demo_dir / camera
            if not camera_dir.exists():
                continue
            
            pcd_files = sorted(camera_dir.glob("frame_*.pcd"))
            
            # Sample frames
            n_sample = max(1, int(len(pcd_files) * sample_rate))
            sampled_files = np.random.choice(pcd_files, n_sample, replace=False) if len(pcd_files) > 0 else []
            
            for pcd_file in sampled_files:
                try:
                    pcd = o3d.io.read_point_cloud(str(pcd_file))
                    xyz = np.asarray(pcd.points, dtype=np.float32)
                    
                    if len(xyz) > 0:
                        all_xyz_min.append(xyz.min(axis=0))
                        all_xyz_max.append(xyz.max(axis=0))
                        total_frames_processed += 1
                except Exception as e:
                    print(f"Error loading {pcd_file}: {e}")
    
    print(f"\nTotal frames processed: {total_frames_processed}")
    
    if len(all_xyz_min) == 0:
        raise ValueError("No PCD data found!")
    
    # Compute global min/max across all frames
    global_min = np.stack(all_xyz_min).min(axis=0)
    global_max = np.stack(all_xyz_max).max(axis=0)
    
    print("\nXYZ Statistics:")
    print(f"  Min: {global_min}")
    print(f"  Max: {global_max}")
    print(f"  Range: {global_max - global_min}")
    
    stats = {
        "xyz": {
            "min": global_min.tolist(),
            "max": global_max.tolist(),
        }
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Compute PCD XYZ statistics')
    parser.add_argument('--pcd-root', required=True, help='PCD root directory')
    parser.add_argument('--cameras', nargs='+', default=['head', 'left_wrist', 'right_wrist'],
                       help='Camera names')
    parser.add_argument('--output', required=True, help='Output JSON file path')
    parser.add_argument('--sample-rate', type=float, default=0.1,
                       help='Fraction of frames to sample (0.1 = 10%%)')
    
    args = parser.parse_args()
    
    pcd_root = Path(args.pcd_root)
    if not pcd_root.exists():
        raise FileNotFoundError(f"PCD root not found: {pcd_root}")
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = compute_pcd_xyz_stats(pcd_root, args.cameras, args.sample_rate)
    
    with open(output_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Statistics saved to: {output_path}")


if __name__ == '__main__':
    main()
