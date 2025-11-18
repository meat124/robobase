#!/usr/bin/env python3
"""
Visualize dataset timeline to verify qpos and actions are correctly recorded
"""
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

hdf5_path = '../data/demonstrations/0.9.0/SaucepanToHob.hdf5'

print("=" * 80)
print("DATASET TIMELINE VISUALIZATION")
print("=" * 80)

with h5py.File(hdf5_path, 'r') as f:
    demo_id = 'demo_1'
    demo = f['data'][demo_id]
    
    actions = demo['actions'][:]
    prop = demo['obs']['proprioception'][:]
    floating_base_actions = demo['obs']['proprioception_floating_base_actions'][:]
    gripper_obs = demo['obs']['proprioception_grippers'][:]
    
    print(f"\n📊 Loaded {demo_id}: {len(actions)} frames")
    
    # Create a comprehensive visualization
    fig = plt.figure(figsize=(20, 24))
    
    # Use all frames to see full trajectory
    n_frames = len(actions)
    timesteps = np.arange(n_frames)
    
    # ========================================================================
    # 1. MOBILE BASE
    # ========================================================================
    print("\n🎨 Plotting Mobile Base...")
    
    # Plot 1: Mobile Base Position (accumulated)
    ax1 = plt.subplot(8, 2, 1)
    ax1.plot(timesteps, floating_base_actions[:n_frames, 0], label='X position', linewidth=2)
    ax1.plot(timesteps, floating_base_actions[:n_frames, 1], label='Y position', linewidth=2)
    ax1.set_title('Mobile Base - Accumulated Position (Observation)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Timestep')
    ax1.set_ylabel('Position')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Mobile Base Actions (delta)
    ax2 = plt.subplot(8, 2, 2)
    ax2.plot(timesteps, actions[:n_frames, 0], label='X delta', linewidth=2)
    ax2.plot(timesteps, actions[:n_frames, 1], label='Y delta', linewidth=2)
    ax2.set_title('Mobile Base - Delta Actions', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Timestep')
    ax2.set_ylabel('Delta Position')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Mobile Base - Verify Cumsum
    ax3 = plt.subplot(8, 2, 3)
    cumsum_actions = np.cumsum(actions[:n_frames, 0:2], axis=0)
    ax3.plot(timesteps, floating_base_actions[:n_frames, 0], label='Obs X', linewidth=2, alpha=0.7)
    ax3.plot(timesteps, cumsum_actions[:, 0], label='Cumsum Action X', linewidth=2, linestyle='--', alpha=0.7)
    ax3.set_title('Mobile Base X - Verification (Obs vs Cumsum)', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Timestep')
    ax3.set_ylabel('Position')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Mobile Base Yaw
    ax4 = plt.subplot(8, 2, 4)
    ax4.plot(timesteps, floating_base_actions[:n_frames, 2], label='Yaw obs', linewidth=2)
    cumsum_yaw = np.cumsum(actions[:n_frames, 2])
    ax4.plot(timesteps, cumsum_yaw, label='Cumsum yaw action', linewidth=2, linestyle='--', alpha=0.7)
    ax4.set_title('Mobile Base - Yaw', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Timestep')
    ax4.set_ylabel('Angle (rad)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # ========================================================================
    # 2. TORSO
    # ========================================================================
    print("🎨 Plotting Torso...")
    
    torso_qpos = prop[:n_frames, 27]
    torso_actions = actions[:n_frames, 3]
    
    # Plot 5: Torso Position
    ax5 = plt.subplot(8, 2, 5)
    ax5.plot(timesteps, torso_qpos, label='qpos (observation)', linewidth=2, color='blue')
    ax5.set_title('Torso - Position (Observation)', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Timestep')
    ax5.set_ylabel('Position')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Torso Actions
    ax6 = plt.subplot(8, 2, 6)
    ax6.plot(timesteps, torso_actions, label='action (delta)', linewidth=2, color='red')
    ax6.set_title('Torso - Action (Delta)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Timestep')
    ax6.set_ylabel('Delta Position')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Torso - Verify Delta
    ax7 = plt.subplot(8, 2, 7)
    ax7.plot(timesteps[1:], torso_qpos[1:], label='Actual qpos[t]', linewidth=2, alpha=0.7)
    predicted_qpos = torso_qpos[:-1] + torso_actions[:-1]
    ax7.plot(timesteps[1:], predicted_qpos, label='qpos[t-1] + action[t-1]', linewidth=2, linestyle='--', alpha=0.7)
    ax7.set_title('Torso - Delta Verification', fontsize=12, fontweight='bold')
    ax7.set_xlabel('Timestep')
    ax7.set_ylabel('Position')
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Torso - Prediction Error
    ax8 = plt.subplot(8, 2, 8)
    error = np.abs(torso_qpos[1:] - predicted_qpos)
    ax8.plot(timesteps[1:], error, linewidth=2, color='red')
    ax8.set_title(f'Torso - Prediction Error (mean={error.mean():.6f})', fontsize=12, fontweight='bold')
    ax8.set_xlabel('Timestep')
    ax8.set_ylabel('Absolute Error')
    ax8.grid(True, alpha=0.3)
    
    # ========================================================================
    # 3. LEFT ARM
    # ========================================================================
    print("🎨 Plotting Left Arm...")
    
    # Left arm has non-consecutive indices: [0, 1, 2, 3, 12]
    left_arm_qpos = np.concatenate([prop[:n_frames, 0:4], prop[:n_frames, 12:13]], axis=1)
    left_arm_actions = actions[:n_frames, 4:9]
    
    # Plot 9: Left Arm Position (all joints)
    ax9 = plt.subplot(8, 2, 9)
    for i in range(5):
        ax9.plot(timesteps, left_arm_qpos[:, i], label=f'Joint {i}', linewidth=1.5, alpha=0.8)
    ax9.set_title('Left Arm - Joint Positions (Observation)', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Timestep')
    ax9.set_ylabel('Position (rad)')
    ax9.legend(loc='best', fontsize=8)
    ax9.grid(True, alpha=0.3)
    
    # Plot 10: Left Arm Actions (all joints)
    ax10 = plt.subplot(8, 2, 10)
    for i in range(5):
        ax10.plot(timesteps, left_arm_actions[:, i], label=f'Joint {i}', linewidth=1.5, alpha=0.8)
    ax10.set_title('Left Arm - Actions', fontsize=12, fontweight='bold')
    ax10.set_xlabel('Timestep')
    ax10.set_ylabel('Action')
    ax10.legend(loc='best', fontsize=8)
    ax10.grid(True, alpha=0.3)
    
    # ========================================================================
    # 4. RIGHT ARM
    # ========================================================================
    print("🎨 Plotting Right Arm...")
    
    # Right arm has non-consecutive indices: [13, 14, 15, 16, 25]
    right_arm_qpos = np.concatenate([prop[:n_frames, 13:17], prop[:n_frames, 25:26]], axis=1)
    right_arm_actions = actions[:n_frames, 9:14]  # actions[9:14] is right arm
    
    # Plot 11: Right Arm Position
    ax11 = plt.subplot(8, 2, 11)
    for i in range(5):
        ax11.plot(timesteps, right_arm_qpos[:, i], label=f'Joint {i}', linewidth=1.5, alpha=0.8)
    ax11.set_title('Right Arm - Joint Positions (Observation)', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Timestep')
    ax11.set_ylabel('Position (rad)')
    ax11.legend(loc='best', fontsize=8)
    ax11.grid(True, alpha=0.3)
    
    # Plot 12: Right Arm Actions
    ax12 = plt.subplot(8, 2, 12)
    for i in range(5):
        ax12.plot(timesteps, right_arm_actions[:, i], label=f'Joint {i}', linewidth=1.5, alpha=0.8)
    ax12.set_title('Right Arm - Actions', fontsize=12, fontweight='bold')
    ax12.set_xlabel('Timestep')
    ax12.set_ylabel('Action')
    ax12.legend(loc='best', fontsize=8)
    ax12.grid(True, alpha=0.3)
    
    # ========================================================================
    # 5. GRIPPERS
    # ========================================================================
    print("🎨 Plotting Grippers...")
    
    left_gripper_obs = gripper_obs[:n_frames, 0]
    right_gripper_obs = gripper_obs[:n_frames, 1]
    left_gripper_action = actions[:n_frames, 14]  # Actual left gripper index
    right_gripper_action = actions[:n_frames, 15]
    
    # Plot 13: Gripper Observations
    ax13 = plt.subplot(8, 2, 13)
    ax13.plot(timesteps, left_gripper_obs, label='Left gripper', linewidth=2, marker='o', markersize=2)
    ax13.plot(timesteps, right_gripper_obs, label='Right gripper', linewidth=2, marker='s', markersize=2)
    ax13.set_title('Grippers - Observations', fontsize=12, fontweight='bold')
    ax13.set_xlabel('Timestep')
    ax13.set_ylabel('Gripper State (0=open, 1=closed)')
    ax13.legend()
    ax13.grid(True, alpha=0.3)
    
    # Plot 14: Gripper Actions
    ax14 = plt.subplot(8, 2, 14)
    ax14.plot(timesteps, left_gripper_action, label='Left gripper', linewidth=2, marker='o', markersize=2)
    ax14.plot(timesteps, right_gripper_action, label='Right gripper', linewidth=2, marker='s', markersize=2)
    ax14.set_title('Grippers - Actions', fontsize=12, fontweight='bold')
    ax14.set_xlabel('Timestep')
    ax14.set_ylabel('Gripper Action')
    ax14.legend()
    ax14.grid(True, alpha=0.3)
    
    # Plot 15: Gripper Correlation Check (Left)
    ax15 = plt.subplot(8, 2, 15)
    ax15.plot(timesteps[1:], left_gripper_obs[1:], label='Obs[t]', linewidth=2, alpha=0.7)
    ax15.plot(timesteps[:-1], left_gripper_action[:-1], label='Action[t-1]', linewidth=2, linestyle='--', alpha=0.7)
    corr_left = np.corrcoef(left_gripper_obs[1:], left_gripper_action[:-1])[0, 1]
    ax15.set_title(f'Left Gripper - Obs vs Action (corr={corr_left:.3f})', fontsize=12, fontweight='bold')
    ax15.set_xlabel('Timestep')
    ax15.set_ylabel('Value')
    ax15.legend()
    ax15.grid(True, alpha=0.3)
    
    # Plot 16: Gripper Correlation Check (Right)
    ax16 = plt.subplot(8, 2, 16)
    ax16.plot(timesteps[1:], right_gripper_obs[1:], label='Obs[t]', linewidth=2, alpha=0.7)
    ax16.plot(timesteps[:-1], right_gripper_action[:-1], label='Action[t-1]', linewidth=2, linestyle='--', alpha=0.7)
    corr_right = np.corrcoef(right_gripper_obs[1:], right_gripper_action[:-1])[0, 1]
    ax16.set_title(f'Right Gripper - Obs vs Action (corr={corr_right:.3f})', fontsize=12, fontweight='bold')
    ax16.set_xlabel('Timestep')
    ax16.set_ylabel('Value')
    ax16.legend()
    ax16.grid(True, alpha=0.3)
    
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
    
    print("\n✅ Mobile Base:")
    cumsum_x = np.cumsum(actions[:, 0])
    corr_x = np.corrcoef(floating_base_actions[:, 0], cumsum_x)[0, 1]
    print(f"   X: obs vs cumsum(action) correlation = {corr_x:.6f}")
    if corr_x > 0.99:
        print(f"   ✅ PERFECT - Observation is accumulated actions")
    
    print("\n✅ Torso:")
    torso_qpos = prop[:, 27]
    torso_actions = actions[:, 3]
    predicted = torso_qpos[:-1] + torso_actions[:-1]
    error = np.abs(torso_qpos[1:] - predicted).mean()
    print(f"   Delta prediction error = {error:.6f}")
    if error < 0.01:
        print(f"   ✅ GOOD - Consistent with delta mode")
    
    print("\n✅ Arms:")
    # Left arm: prop[0, 1, 2, 3, 12]
    left_arm_qpos = np.concatenate([prop[:, 0:4], prop[:, 12:13]], axis=1)
    left_arm_actions = actions[:, 4:9]
    # Right arm: prop[13, 14, 15, 16, 25]
    right_arm_qpos = np.concatenate([prop[:, 13:17], prop[:, 25:26]], axis=1)
    right_arm_actions = actions[:, 9:14]
    
    left_corr = np.mean([np.corrcoef(left_arm_qpos[:, i], left_arm_actions[:, i])[0, 1] for i in range(5)])
    right_corr = np.mean([np.corrcoef(right_arm_qpos[:, i], right_arm_actions[:, i])[0, 1] for i in range(5)])
    
    print(f"   Left arm:  avg correlation = {left_corr:.3f}")
    print(f"   Right arm: avg correlation = {right_corr:.3f}")
    if left_corr > 0.9:
        print(f"   ✅ Left arm correctly aligned (absolute position mode)")
    if right_corr > 0.9:
        print(f"   ✅ Right arm correctly aligned (absolute position mode)")
    
    print("\n✅ Grippers:")
    left_corr = np.corrcoef(gripper_obs[1:, 0], actions[:-1, 14])[0, 1]
    right_corr = np.corrcoef(gripper_obs[1:, 1], actions[:-1, 15])[0, 1]
    print(f"   Left gripper:  obs[t] vs action[t-1] correlation = {left_corr:.3f}")
    print(f"   Right gripper: obs[t] vs action[t-1] correlation = {right_corr:.3f}")
    if right_corr > 0.9:
        print(f"   ✅ Right gripper correctly aligned")
    if left_corr > 0.9:
        print(f"   ✅ Left gripper correctly aligned")

print("\n" + "=" * 80)
print("📊 Visualization saved to: dataset_timeline_visualization.png")
print("=" * 80)
