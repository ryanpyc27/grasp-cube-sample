"""
Test script for LiftCubeSO101-v1 environment
Records a video to visualize the camera angles and environment behavior
"""
import gymnasium as gym
import numpy as np
import imageio
import os
from pathlib import Path

# Import the custom environment
import grasp_cube.envs.tasks.lift_cube_so101

def test_lift_cube_env(
    num_episodes=3,
    max_steps_per_episode=50,
    video_path="test_videos/lift_cube_test.mp4",
    num_envs=1,
    render_mode="rgb_array"
):
    """
    Test the LiftCubeSO101-v1 environment and record video
    
    Args:
        num_episodes: Number of episodes to run
        max_steps_per_episode: Maximum steps per episode
        video_path: Path to save the video
        num_envs: Number of parallel environments (1 for single env)
        render_mode: Render mode ('rgb_array' for video recording)
    """
    # Create output directory
    Path(video_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"Creating LiftCubeSO101-v1 environment...")
    
    # Create environment
    if num_envs == 1:
        # Single environment (CPU mode for easier video recording)
        env = gym.make(
            "LiftCubeSO101-v1",
            obs_mode="state",
            render_mode=render_mode,
            control_mode="pd_joint_delta_pos",
        )
    else:
        # Vectorized environment (GPU mode)
        env = gym.make(
            "LiftCubeSO101-v1",
            obs_mode="state",
            render_mode=render_mode,
            control_mode="pd_joint_delta_pos",
            num_envs=num_envs,
        )
    
    print(f"Environment created successfully!")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Video writer
    frames = []
    
    for episode in range(num_episodes):
        print(f"\n=== Episode {episode + 1}/{num_episodes} ===")
        obs, info = env.reset()
        
        episode_reward = 0
        episode_success = False
        
        for step in range(max_steps_per_episode):
            # Random action for testing
            action = env.action_space.sample()
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            if num_envs == 1:
                episode_reward += reward if isinstance(reward, (int, float)) else reward.item()
                done = terminated or truncated
                success = info.get("success", False)
                reward_value = reward if isinstance(reward, (int, float)) else reward.item()
            else:
                episode_reward += reward[0].item()
                done = terminated[0] or truncated[0]
                success = info.get("success", [False])[0]
                reward_value = reward[0].item()
            
            # Render frame
            frame = env.render()
            if frame is not None:
                if num_envs > 1:
                    # For vectorized env, take first environment's frame
                    if isinstance(frame, np.ndarray) and len(frame.shape) == 4:
                        frame = frame[0]
                
                # Debug: print frame shape for first frame
                if len(frames) == 0:
                    import torch
                    if isinstance(frame, torch.Tensor):
                        print(f"  Frame shape (tensor): {frame.shape}, dtype: {frame.dtype}")
                    else:
                        print(f"  Frame shape (numpy): {frame.shape}, dtype: {frame.dtype}")
                
                frames.append(frame)
            
            if step % 10 == 0:
                print(f"  Step {step}: reward={reward_value:.3f}, success={success}")
            
            if done:
                episode_success = success
                print(f"  Episode finished at step {step}")
                break
        
        print(f"Episode {episode + 1} - Total Reward: {episode_reward:.3f}, Success: {episode_success}")
    
    env.close()
    
    # Save video
    if frames:
        print(f"\nSaving video to {video_path}...")
        # Convert frames to uint8 numpy arrays if needed
        processed_frames = []
        for i, frame in enumerate(frames):
            # Convert torch tensor to numpy if needed
            import torch
            if isinstance(frame, torch.Tensor):
                frame = frame.cpu().numpy()
            
            # Check and fix frame dimensions
            # Expected: (H, W, C) where C is 1, 3, or 4
            if len(frame.shape) == 4:
                # Remove batch dimension if present
                frame = frame[0]
            
            if len(frame.shape) == 3:
                # Check if it's (C, H, W) format, convert to (H, W, C)
                if frame.shape[0] in [1, 3, 4] and frame.shape[0] < frame.shape[1]:
                    frame = np.transpose(frame, (1, 2, 0))
            
            # Convert to uint8 if needed
            if frame.dtype != np.uint8:
                # If values are in [0, 1], scale to [0, 255]
                if frame.max() <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # Debug first frame
            if i == 0:
                print(f"  Processed frame shape: {frame.shape}, dtype: {frame.dtype}")
            
            processed_frames.append(frame)
        
        imageio.mimsave(video_path, processed_frames, fps=20)
        print(f"Video saved! Total frames: {len(processed_frames)}")
        print(f"Video location: {os.path.abspath(video_path)}")
    else:
        print("No frames captured. Check render_mode setting.")
    
    return episode_reward, episode_success


def test_camera_angles():
    """
    Test different camera angles by rendering from different viewpoints
    """
    print("\n" + "="*60)
    print("Testing camera angles...")
    print("="*60)
    
    env = gym.make(
        "LiftCubeSO101-v1",
        obs_mode="state",
        render_mode="rgb_array",
        control_mode="pd_joint_delta_pos",
    )
    
    print("\nEnvironment camera configurations:")
    print(f"  Sensor camera eye: {env.unwrapped.sensor_cam_eye_pos}")
    print(f"  Sensor camera target: {env.unwrapped.sensor_cam_target_pos}")
    print(f"  Human render camera eye: {env.unwrapped.human_cam_eye_pos}")
    print(f"  Human render camera target: {env.unwrapped.human_cam_target_pos}")
    
    env.close()


if __name__ == "__main__":
    print("="*60)
    print("LiftCubeSO101-v1 Environment Test")
    print("="*60)
    
    # Test camera configuration
    test_camera_angles()
    
    # Run environment test with video recording
    print("\n" + "="*60)
    print("Running environment test...")
    print("="*60)
    
    test_lift_cube_env(
        num_episodes=2,
        max_steps_per_episode=50,
        video_path="test_videos/lift_cube_test.mp4",
        num_envs=1,
    )
    
    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)
