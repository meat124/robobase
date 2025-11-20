#!/usr/bin/env python3
"""
Visualize dataset timeline to verify observations and actions are correctly matched
Specifically verifies:
1. Mobile base: observation (qvel velocity) vs action (velocity from delta/dt)
2. Torso: observation (accumulated position) vs action (absolute position)
3. Arms/Grippers: observation (qpos) vs action (absolute position)
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hdf5_path = '../data/demonstrations/0.9.0/SaucepanToHob_successful.hdf5'
dt = 0.02  # Control timestep (50Hz)

print("=" * 80)
print("DATASET TIMELINE VISUALIZATION")
print("=" * 80)

with h5py.File(hdf5_path, 'r') as f:
    demo_id = 'demo_1'
    demo = f['data'][demo_id]
    
    actions = demo['actions'][:]
    prop = demo['obs']['proprioception'][:]
    floating_base_actions = demo['obs']['proprioception_floating_base_actions'][:]
    floating_base = demo['obs']['proprioception_floating_base'][:]
    gripper_obs = demo['obs']['proprioception_grippers'][:]
    
    print(f"\n📊 Loaded {demo_id}: {len(actions)} frames")
    print(f"   - Actions shape: {actions.shape}")
    print(f"   - Proprioception shape: {prop.shape}")
    print(f"   - Floating base actions shape: {floating_base_actions.shape}")
    print(f"   - Gripper obs shape: {gripper_obs.shape}")
    
    # Compute mobile base velocity from position changes using proprioception_floating_base
    mobile_base_pos_diffs = np.diff(floating_base[:, [0, 1, 3]], axis=0)  # X, Y, RZ
    mobile_base_vel_obs = mobile_base_pos_diffs / dt
    # Prepend zero velocity for first frame
    mobile_base_vel_obs = np.concatenate([np.zeros((1, 3)), mobile_base_vel_obs], axis=0)
    
    # Create a comprehensive visualization
    fig = plt.figure(figsize=(20, 28))
    
    # Use all frames to see full trajectory
    n_frames = len(actions)
    timesteps = np.arange(n_frames)
    
    # ========================================================================
    # 1. MOBILE BASE - VELOCITY MODE
    # ========================================================================
    print("\n🎨 Plotting Mobile Base (Velocity Mode)...")
    
    # Mobile base velocity observations are computed from position changes (using proprioception_floating_base)
    # Use proprioception_floating_base directly (NOT floating_base_actions)
    mobile_base_pos_diffs = np.diff(floating_base[:, [0, 1, 3]], axis=0)  # X, Y, RZ from proprioception_floating_base
    mobile_base_vel_obs = mobile_base_pos_diffs / dt
    # Prepend zero velocity for first frame
    mobile_base_vel_obs = np.concatenate([np.zeros((1, 3)), mobile_base_vel_obs], axis=0)
    
    # Convert actions (deltas) to velocity: delta / dt
    mobile_base_vel_actions = np.concatenate([
        actions[:n_frames, 0:2] / dt,  # X, Y velocity
        actions[:n_frames, 3:4] / dt   # RZ velocity
    ], axis=-1)
    
    # Plot 1: Mobile Base Velocity Observation (from position changes)
    ax1 = plt.subplot(9, 2, 1)
    ax1.plot(timesteps, mobile_base_vel_obs[:n_frames, 0], label='X vel (from floating_base)', linewidth=2, alpha=0.8)
    ax1.plot(timesteps, mobile_base_vel_obs[:n_frames, 1], label='Y vel (from floating_base)', linewidth=2, alpha=0.8)
    ax1.set_title('Mobile Base Velocity - Observation (from proprioception_floating_base)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Velocity (m/s)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mobile Base Velocity Action (delta/dt)
    ax2 = plt.subplot(9, 2, 2)
    ax2.plot(timesteps, mobile_base_vel_actions[:, 0], label='X vel (delta/dt)', linewidth=2, alpha=0.8)
    ax2.plot(timesteps, mobile_base_vel_actions[:, 1], label='Y vel (delta/dt)', linewidth=2, alpha=0.8)
    ax2.set_title('Mobile Base Velocity - Action (delta/dt)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mobile Base X Velocity - Overlay Comparison
    ax3 = plt.subplot(9, 2, 3)
    ax3.plot(timesteps, mobile_base_vel_obs[:n_frames, 0], label='Obs (from floating_base)', linewidth=2, alpha=0.7)
    ax3.plot(timesteps, mobile_base_vel_actions[:, 0], label='Action (delta/dt)', linewidth=2, linestyle='--', alpha=0.7)
    corr_x = np.corrcoef(mobile_base_vel_obs[:n_frames, 0], mobile_base_vel_actions[:, 0])[0, 1]
    rmse_x = np.sqrt(np.mean((mobile_base_vel_obs[:n_frames, 0] - mobile_base_vel_actions[:, 0])**2))
    ax3.set_title(f'Mobile Base X Velocity Match (corr={corr_x:.4f}, RMSE={rmse_x:.4f})', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mobile Base RZ Velocity - Overlay Comparison
    ax4 = plt.subplot(9, 2, 4)
    ax4.plot(timesteps, mobile_base_vel_obs[:n_frames, 2], label='Obs (from floating_base)', linewidth=2, alpha=0.7)
    ax4.plot(timesteps, mobile_base_vel_actions[:, 2], label='Action (delta/dt)', linewidth=2, linestyle='--', alpha=0.7)
    corr_rz = np.corrcoef(mobile_base_vel_obs[:n_frames, 2], mobile_base_vel_actions[:, 2])[0, 1]
    rmse_rz = np.sqrt(np.mean((mobile_base_vel_obs[:n_frames, 2] - mobile_base_vel_actions[:, 2])**2))
    ax4.set_title(f'Mobile Base RZ Velocity Match (corr={corr_rz:.4f}, RMSE={rmse_rz:.4f})', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Angular Velocity (rad/s)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # 2. TORSO (Z - pelvis_z) - ABSOLUTE POSITION MODE
    # ========================================================================
    print("🎨 Plotting Torso (pelvis_z - Absolute Position Mode)...")
    
    # Torso observation: direct from proprioception_floating_base (NOT accumulated from deltas)
    torso_obs = floating_base[:n_frames, 2]  # Z from proprioception_floating_base (direct measurement)
    
    # Torso action: convert deltas to absolute positions
    # Start from the first observed torso position and accumulate deltas
    torso_deltas = actions[:n_frames, 2]  # Delta Z from actions
    # Simulate what dataset loader does: start from obs[0] and cumsum deltas
    torso_actions_absolute = np.zeros_like(torso_deltas)
    for i in range(len(torso_deltas)):
        if i == 0:
            # First action: current position + first delta
            torso_actions_absolute[i] = torso_obs[0] + torso_deltas[i]
        else:
            # Subsequent actions: previous action + current delta
            torso_actions_absolute[i] = torso_actions_absolute[i-1] + torso_deltas[i]
    
    # Plot 5: Torso Position Observation (accumulated)
    ax5 = plt.subplot(9, 2, 5)
    ax5.plot(timesteps, torso_obs, label='Obs (from floating_base)', linewidth=2, color='blue')
    ax5.set_title('Torso (pelvis_z) - Observation (Direct from proprioception_floating_base)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Position (m)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Torso Actions (absolute positions)
    ax6 = plt.subplot(9, 2, 6)
    ax6.plot(timesteps, torso_actions_absolute, label='Action (absolute)', linewidth=2, color='red')
    ax6.set_title('Torso (pelvis_z) - Action (Absolute Position)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Position (m)')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Torso - Overlay Comparison (Obs vs Action)
    ax7 = plt.subplot(9, 2, 7)
    ax7.plot(timesteps, torso_obs, label='Obs (from floating_base)', linewidth=2, alpha=0.7)
    ax7.plot(timesteps, torso_actions_absolute, label='Action (absolute)', linewidth=2, linestyle='--', alpha=0.7)
    corr_z = np.corrcoef(torso_obs, torso_actions_absolute)[0, 1]
    rmse_z = np.sqrt(np.mean((torso_obs - torso_actions_absolute)**2))
    ax7.set_title(f'Torso Position Match (corr={corr_z:.4f}, RMSE={rmse_z:.4f})', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Timestep')
    ax7.set_ylabel('Position (m)')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Torso - Match Error Over Time
    ax8 = plt.subplot(9, 2, 8)
    error = np.abs(torso_obs - torso_actions_absolute)
    ax8.plot(timesteps, error, linewidth=2, color='red')
    ax8.set_title(f'Torso Position Error (mean={error.mean():.6f}, max={error.max():.6f})', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Timestep')
    ax8.set_ylabel('Absolute Error (m)')
    ax8.grid(True, alpha=0.3)
    
    # ========================================================================
    # 3. LEFT ARM - ABSOLUTE POSITION MODE
    # ========================================================================
    print("🎨 Plotting Left Arm (Absolute Position Mode)...")
    
    # Left arm has non-consecutive indices: [0, 1, 2, 3, 12]
    left_arm_qpos = np.concatenate([prop[:n_frames, 0:4], prop[:n_frames, 12:13]], axis=1)
    left_arm_actions = actions[:n_frames, 4:9]
    
    # Plot 9: Left Arm Position (all joints) - Observation
    ax9 = plt.subplot(9, 2, 9)
    for i in range(5):
        ax9.plot(timesteps, left_arm_qpos[:, i], label=f'Joint {i}', linewidth=1.5, alpha=0.8)
    ax9.set_title('Left Arm - Joint Positions (Observation from qpos)', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Timestep')
    ax9.set_ylabel('Position (rad)')
    ax9.legend(loc='best', fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    # Plot 10: Left Arm Actions (all joints) - Absolute Positions
    ax10 = plt.subplot(9, 2, 10)
    for i in range(5):
        ax10.plot(timesteps, left_arm_actions[:, i], label=f'Joint {i}', linewidth=1.5, alpha=0.8)
    ax10.set_title('Left Arm - Actions (Absolute Positions)', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Timestep')
    ax10.set_ylabel('Action')
    ax10.legend(loc='best', fontsize=8)
    ax10.grid(True, alpha=0.3)
    
    # Plot 11: Left Arm - Correlation Analysis
    ax11 = plt.subplot(9, 2, 11)
    correlations = [np.corrcoef(left_arm_qpos[:, i], left_arm_actions[:, i])[0, 1] for i in range(5)]
    ax11.bar(range(5), correlations, alpha=0.7)
    ax11.axhline(y=0.9, color='g', linestyle='--', label='Target (0.9)')
    ax11.set_title(f'Left Arm - Obs vs Action Correlation (avg={np.mean(correlations):.3f})', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Joint Index')
    ax11.set_ylabel('Correlation')
    ax11.set_ylim([0, 1.05])
    ax11.legend()
    ax11.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # 4. RIGHT ARM - ABSOLUTE POSITION MODE
    # ========================================================================
    print("🎨 Plotting Right Arm (Absolute Position Mode)...")
    
    # Right arm has non-consecutive indices: [13, 14, 15, 16, 25]
    right_arm_qpos = np.concatenate([prop[:n_frames, 13:17], prop[:n_frames, 25:26]], axis=1)
    right_arm_actions = actions[:n_frames, 9:14]  # actions[9:14] is right arm
    
    # Plot 12: Right Arm Position - Observation
    ax12 = plt.subplot(9, 2, 12)
    for i in range(5):
        ax12.plot(timesteps, right_arm_qpos[:, i], label=f'Joint {i}', linewidth=1.5, alpha=0.8)
    ax12.set_title('Right Arm - Joint Positions (Observation from qpos)', fontsize=12, fontweight='bold')
    ax12.set_xlabel('Timestep')
    ax12.set_ylabel('Position (rad)')
    ax12.legend(loc='best', fontsize=8)
    ax12.grid(True, alpha=0.3)
    
    # Plot 13: Right Arm Actions - Absolute Positions
    ax13 = plt.subplot(9, 2, 13)
    for i in range(5):
        ax13.plot(timesteps, right_arm_actions[:, i], label=f'Joint {i}', linewidth=1.5, alpha=0.8)
    ax13.set_title('Right Arm - Actions (Absolute Positions)', fontsize=12, fontweight='bold')
    ax13.set_xlabel('Timestep')
    ax13.set_ylabel('Action')
    ax13.legend(loc='best', fontsize=8)
    ax13.grid(True, alpha=0.3)
    
    # Plot 14: Right Arm - Correlation Analysis
    ax14 = plt.subplot(9, 2, 14)
    correlations = [np.corrcoef(right_arm_qpos[:, i], right_arm_actions[:, i])[0, 1] for i in range(5)]
    ax14.bar(range(5), correlations, alpha=0.7, color='orange')
    ax14.axhline(y=0.9, color='g', linestyle='--', label='Target (0.9)')
    ax14.set_title(f'Right Arm - Obs vs Action Correlation (avg={np.mean(correlations):.3f})', fontsize=12, fontweight='bold')
    ax14.set_xlabel('Joint Index')
    ax14.set_ylabel('Correlation')
    ax14.set_ylim([0, 1.05])
    ax14.legend()
    ax14.grid(True, alpha=0.3, axis='y')
    
    # ========================================================================
    # 5. GRIPPERS - ABSOLUTE POSITION MODE
    # ========================================================================
    print("🎨 Plotting Grippers (Absolute Position Mode)...")
    
    left_gripper_obs = gripper_obs[:n_frames, 0]
    right_gripper_obs = gripper_obs[:n_frames, 1]
    left_gripper_action = actions[:n_frames, 14]  # Left gripper at index 14
    right_gripper_action = actions[:n_frames, 15]  # Right gripper at index 15
    
    # Plot 15: Gripper Observations
    ax15 = plt.subplot(9, 2, 15)
    ax15.plot(timesteps, left_gripper_obs, label='Left gripper obs', linewidth=2, marker='o', markersize=2)
    ax15.plot(timesteps, right_gripper_obs, label='Right gripper obs', linewidth=2, marker='s', markersize=2)
    ax15.set_title('Grippers - Observations (0=open, 1=closed)', fontsize=12, fontweight='bold')
    ax15.set_xlabel('Timestep')
    ax15.set_ylabel('Gripper State')
    ax15.legend()
    ax15.grid(True, alpha=0.3)
    
    # Plot 16: Gripper Actions
    ax16 = plt.subplot(9, 2, 16)
    ax16.plot(timesteps, left_gripper_action, label='Left gripper action', linewidth=2, marker='o', markersize=2)
    ax16.plot(timesteps, right_gripper_action, label='Right gripper action', linewidth=2, marker='s', markersize=2)
    ax16.set_title('Grippers - Actions (Absolute Positions)', fontsize=12, fontweight='bold')
    ax16.set_xlabel('Timestep')
    ax16.set_ylabel('Gripper Action')
    ax16.legend()
    ax16.grid(True, alpha=0.3)
    
    # Plot 17: Left Gripper - Overlay Comparison
    ax17 = plt.subplot(9, 2, 17)
    ax17.plot(timesteps, left_gripper_obs, label='Obs', linewidth=2, alpha=0.7)
    ax17.plot(timesteps, left_gripper_action, label='Action', linewidth=2, linestyle='--', alpha=0.7)
    corr_left = np.corrcoef(left_gripper_obs, left_gripper_action)[0, 1]
    rmse_left = np.sqrt(np.mean((left_gripper_obs - left_gripper_action)**2))
    ax17.set_title(f'Left Gripper Match (corr={corr_left:.3f}, RMSE={rmse_left:.3f})', fontsize=12, fontweight='bold')
    ax17.set_xlabel('Timestep')
    ax17.set_ylabel('Value')
    ax17.legend()
    ax17.grid(True, alpha=0.3)
    
    # Plot 18: Right Gripper - Overlay Comparison
    ax18 = plt.subplot(9, 2, 18)
    ax18.plot(timesteps, right_gripper_obs, label='Obs', linewidth=2, alpha=0.7)
    ax18.plot(timesteps, right_gripper_action, label='Action', linewidth=2, linestyle='--', alpha=0.7)
    corr_right = np.corrcoef(right_gripper_obs, right_gripper_action)[0, 1]
    rmse_right = np.sqrt(np.mean((right_gripper_obs - right_gripper_action)**2))
    ax18.set_title(f'Right Gripper Match (corr={corr_right:.3f}, RMSE={rmse_right:.3f})', fontsize=12, fontweight='bold')
    ax18.set_xlabel('Timestep')
    ax18.set_ylabel('Value')
    ax18.legend()
    ax18.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dataset_timeline_visualization.png', dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved: dataset_timeline_visualization.png")

print("\n" + "=" * 80)
print("VERIFICATION SUMMARY")
print("=" * 80)

with h5py.File(hdf5_path, 'r') as f:
    demo = f['data']['demo_1']
    
    actions = demo['actions'][:]
    prop = demo['obs']['proprioception'][:]
    floating_base_actions = demo['obs']['proprioception_floating_base_actions'][:]
    gripper_obs = demo['obs']['proprioception_grippers'][:]
    
    print("\n✅ MOBILE BASE (Velocity Mode):")
    print("   Observation: diff(proprioception_floating_base) / dt (velocity from position changes)")
    print("   Action: delta / dt (converted to velocity)")
    
    # Compute velocities from proprioception_floating_base
    mobile_base_pos_diffs = np.diff(floating_base[:, [0, 1, 3]], axis=0)
    mobile_base_vel_obs = np.concatenate([np.zeros((1, 3)), mobile_base_pos_diffs / dt], axis=0)
    mobile_base_vel_actions = np.concatenate([
        actions[:, 0:2] / dt,
        actions[:, 3:4] / dt
    ], axis=-1)
    
    corr_x = np.corrcoef(mobile_base_vel_obs[:, 0], mobile_base_vel_actions[:, 0])[0, 1]
    corr_y = np.corrcoef(mobile_base_vel_obs[:, 1], mobile_base_vel_actions[:, 1])[0, 1]
    corr_rz = np.corrcoef(mobile_base_vel_obs[:, 2], mobile_base_vel_actions[:, 2])[0, 1]
    
    rmse_x = np.sqrt(np.mean((mobile_base_vel_obs[:, 0] - mobile_base_vel_actions[:, 0])**2))
    rmse_y = np.sqrt(np.mean((mobile_base_vel_obs[:, 1] - mobile_base_vel_actions[:, 1])**2))
    rmse_rz = np.sqrt(np.mean((mobile_base_vel_obs[:, 2] - mobile_base_vel_actions[:, 2])**2))
    
    print(f"   X velocity:  corr={corr_x:.4f}, RMSE={rmse_x:.4f} m/s")
    print(f"   Y velocity:  corr={corr_y:.4f}, RMSE={rmse_y:.4f} m/s")
    print(f"   RZ velocity: corr={corr_rz:.4f}, RMSE={rmse_rz:.4f} rad/s")
    
    if corr_x > 0.95 and corr_y > 0.95 and corr_rz > 0.95:
        print(f"   ✅ EXCELLENT - Very high correlation, perfect match!")
    elif corr_x > 0.7 and corr_y > 0.7 and corr_rz > 0.7:
        print(f"   ✅ GOOD - Observations and actions show strong correlation")
    else:
        print(f"   ⚠️  WARNING - Low correlation detected")
    
    print("\n✅ TORSO (Absolute Position Mode):")
    print("   Observation: direct from proprioception_floating_base[:, 2] (pelvis_z)")
    print("   Action: current_pos + cumsum(deltas) = absolute position")
    
    torso_obs = floating_base[:, 2]  # Direct from proprioception_floating_base
    torso_deltas = actions[:, 2]
    
    # Simulate dataset processing
    torso_actions_absolute = np.zeros_like(torso_deltas)
    for i in range(len(torso_deltas)):
        if i == 0:
            torso_actions_absolute[i] = torso_obs[0] + torso_deltas[i]
        else:
            torso_actions_absolute[i] = torso_actions_absolute[i-1] + torso_deltas[i]
    
    corr_z = np.corrcoef(torso_obs, torso_actions_absolute)[0, 1]
    rmse_z = np.sqrt(np.mean((torso_obs - torso_actions_absolute)**2))
    error_mean = np.abs(torso_obs - torso_actions_absolute).mean()
    error_max = np.abs(torso_obs - torso_actions_absolute).max()
    
    print(f"   Position correlation: {corr_z:.4f}")
    print(f"   RMSE: {rmse_z:.4f} m")
    print(f"   Mean absolute error: {error_mean:.6f} m")
    print(f"   Max absolute error: {error_max:.6f} m")
    
    if corr_z > 0.95:
        print(f"   ✅ EXCELLENT - Very high correlation")
    elif error_mean < 0.01:
        print(f"   ✅ GOOD - Low error")
    else:
        print(f"   ⚠️  WARNING - Check torso processing")
    
    print("\n✅ ARMS (Absolute Position Mode):")
    print("   Observation: qpos (joint positions)")
    print("   Action: absolute joint positions")
    
    # Left arm: prop[0, 1, 2, 3, 12]
    left_arm_qpos = np.concatenate([prop[:, 0:4], prop[:, 12:13]], axis=1)
    left_arm_actions = actions[:, 4:9]
    # Right arm: prop[13, 14, 15, 16, 25]
    right_arm_qpos = np.concatenate([prop[:, 13:17], prop[:, 25:26]], axis=1)
    right_arm_actions = actions[:, 9:14]
    
    left_corr = np.mean([np.corrcoef(left_arm_qpos[:, i], left_arm_actions[:, i])[0, 1] for i in range(5)])
    right_corr = np.mean([np.corrcoef(right_arm_qpos[:, i], right_arm_actions[:, i])[0, 1] for i in range(5)])
    
    left_rmse = np.mean([np.sqrt(np.mean((left_arm_qpos[:, i] - left_arm_actions[:, i])**2)) for i in range(5)])
    right_rmse = np.mean([np.sqrt(np.mean((right_arm_qpos[:, i] - right_arm_actions[:, i])**2)) for i in range(5)])
    
    print(f"   Left arm:  avg corr={left_corr:.3f}, avg RMSE={left_rmse:.4f} rad")
    print(f"   Right arm: avg corr={right_corr:.3f}, avg RMSE={right_rmse:.4f} rad")
    
    if left_corr > 0.9 and right_corr > 0.9:
        print(f"   ✅ EXCELLENT - Arms correctly aligned")
    elif left_corr > 0.7 and right_corr > 0.7:
        print(f"   ✅ GOOD - Acceptable alignment")
    else:
        print(f"   ⚠️  WARNING - Check arm processing")
    
    print("\n✅ GRIPPERS (Absolute Position Mode):")
    print("   Observation: gripper positions (0=open, 1=closed)")
    print("   Action: absolute gripper positions")
    
    left_gripper_obs = gripper_obs[:, 0]
    right_gripper_obs = gripper_obs[:, 1]
    left_gripper_action = actions[:, 14]
    right_gripper_action = actions[:, 15]
    
    left_corr = np.corrcoef(left_gripper_obs, left_gripper_action)[0, 1]
    right_corr = np.corrcoef(right_gripper_obs, right_gripper_action)[0, 1]
    
    left_rmse = np.sqrt(np.mean((left_gripper_obs - left_gripper_action)**2))
    right_rmse = np.sqrt(np.mean((right_gripper_obs - right_gripper_action)**2))
    
    print(f"   Left gripper:  corr={left_corr:.3f}, RMSE={left_rmse:.3f}")
    print(f"   Right gripper: corr={right_corr:.3f}, RMSE={right_rmse:.3f}")
    
    if left_corr > 0.9 and right_corr > 0.9:
        print(f"   ✅ EXCELLENT - Grippers correctly aligned")
    elif left_corr > 0.7 or right_corr > 0.7:
        print(f"   ✅ ACCEPTABLE - Some gripper states may vary")
    else:
        print(f"   ⚠️  NOTE - Grippers may have different control characteristics")

print("\n" + "=" * 80)
print("📊 Visualization saved to: dataset_timeline_visualization.png")
print("=" * 80)
