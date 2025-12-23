#!/usr/bin/env python3
"""
Evaluate a trained LeRobot policy in ManiSkill environment.

Usage:
    python eval_policy.py \
        --checkpoint /homes/yichengp/grasp-cube-sample/log/checkpoints/last/pretrained_model \
        --env-id StackCubeSO101-v1 \
        --num-episodes 50 \
        --save-video
"""

import argparse
import gymnasium as gym
import numpy as np
import torch
import imageio
from pathlib import Path
from tqdm import tqdm
from safetensors.torch import load_file

# Import custom environments
import grasp_cube.envs.tasks.stack_cube_so101
import grasp_cube.envs.tasks.sort_cube_so101

# Import lerobot
from lerobot.policies.act.modeling_act import ACTPolicy


def load_action_stats(checkpoint_path: str):
    """Load action normalization statistics from checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    stats_file = checkpoint_path / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    
    if not stats_file.exists():
        print(f"Warning: Stats file not found at {stats_file}")
        return None, None
    
    stats = load_file(str(stats_file))
    action_mean = stats['action.mean'].numpy()
    action_std = stats['action.std'].numpy()
    
    return action_mean, action_std


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LeRobot policy in ManiSkill environment")
    parser.add_argument(
        "--checkpoint", 
        type=str, 
        required=True,
        help="Path to pretrained model directory"
    )
    parser.add_argument(
        "--env-id", 
        type=str, 
        default="StackCubeSO101-v1",
        help="Environment ID"
    )
    parser.add_argument(
        "--num-episodes", 
        type=int, 
        default=50,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--max-steps", 
        type=int, 
        default=600,
        help="Maximum steps per episode"
    )
    parser.add_argument(
        "--save-video", 
        action="store_true",
        help="Save video of evaluation"
    )
    parser.add_argument(
        "--video-dir", 
        type=str, 
        default="eval_videos",
        help="Directory to save videos"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="cuda",
        help="Device to run policy on"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=0,
        help="Random seed"
    )
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Print debug information about observations and actions"
    )
    return parser.parse_args()


def get_observation_for_policy(obs: dict, device: str, debug: bool = False) -> dict:
    """
    Convert ManiSkill observation to LeRobot policy input format.
    
    ManiSkill sensor_data mode provides:
    - obs['sensor_data']['base_camera']['Color'] -> observation.images.image (RGBA float32 [0,1])
    - obs['sensor_data']['wrist_camera']['Color'] -> observation.images.wrist_image (RGBA float32 [0,1])
    - obs['agent']['qpos'] -> observation.state
    
    LeRobot expects (based on training data):
    - observation.images.image: [batch, C, H, W] as float32, values in [0, 1] range
    - observation.images.wrist_image: [batch, C, H, W] as float32, values in [0, 1] range  
    - observation.state: [batch, state_dim]
    
    Note: LeRobot's preprocessor will handle normalization (MEAN_STD) internally.
    The training data was stored as uint8 [0,255] but LeRobot converts to float [0,1] internally.
    """
    policy_obs = {}
    
    # Get images from sensor_data
    # ManiSkill returns images in [B, H, W, C] format as RGBA float32, need to:
    # 1. Take only RGB channels (first 3)
    # 2. Convert to [B, C, H, W]
    if 'sensor_data' in obs:
        # Base camera - 'Color' key contains RGBA float32 [0,1]
        base_image = obs['sensor_data']['base_camera']['Color']
        if isinstance(base_image, np.ndarray):
            base_image = torch.from_numpy(base_image)
        # Take only RGB channels (first 3), ignore Alpha
        base_image = base_image[..., :3]
        # Convert from [B, H, W, C] to [B, C, H, W]
        if base_image.dim() == 4:
            base_image = base_image.permute(0, 3, 1, 2)
        elif base_image.dim() == 3:
            base_image = base_image.permute(2, 0, 1).unsqueeze(0)
        # Clamp to [0, 1] (already float32)
        base_image = torch.clamp(base_image.float(), 0, 1)
        policy_obs['observation.images.image'] = base_image.to(device)
        
        # Wrist camera - 'Color' key contains RGBA float32 [0,1]
        wrist_image = obs['sensor_data']['wrist_camera']['Color']
        if isinstance(wrist_image, np.ndarray):
            wrist_image = torch.from_numpy(wrist_image)
        # Take only RGB channels (first 3), ignore Alpha
        wrist_image = wrist_image[..., :3]
        # Convert from [B, H, W, C] to [B, C, H, W]
        if wrist_image.dim() == 4:
            wrist_image = wrist_image.permute(0, 3, 1, 2)
        elif wrist_image.dim() == 3:
            wrist_image = wrist_image.permute(2, 0, 1).unsqueeze(0)
        # Clamp to [0, 1] (already float32)
        wrist_image = torch.clamp(wrist_image.float(), 0, 1)
        policy_obs['observation.images.wrist_image'] = wrist_image.to(device)
    
    # Get robot state (qpos)
    if 'agent' in obs:
        state = obs['agent']['qpos']
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state)
        # Ensure shape is [B, state_dim]
        if state.dim() == 1:
            state = state.unsqueeze(0)
        policy_obs['observation.state'] = state.float().to(device)
    
    if debug:
        print(f"\n[DEBUG] Observation stats:")
        print(f"  base_image: shape={policy_obs['observation.images.image'].shape}, "
              f"min={policy_obs['observation.images.image'].min():.3f}, "
              f"max={policy_obs['observation.images.image'].max():.3f}")
        print(f"  wrist_image: shape={policy_obs['observation.images.wrist_image'].shape}, "
              f"min={policy_obs['observation.images.wrist_image'].min():.3f}, "
              f"max={policy_obs['observation.images.wrist_image'].max():.3f}")
        print(f"  state: shape={policy_obs['observation.state'].shape}, "
              f"values={policy_obs['observation.state'].cpu().numpy()}")
    
    return policy_obs


def main():
    args = parse_args()
    
    print(f"=" * 60)
    print(f"Evaluating Policy")
    print(f"=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Environment: {args.env_id}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Device: {args.device}")
    print(f"=" * 60)
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load policy
    print("\nLoading policy...")
    policy = ACTPolicy.from_pretrained(args.checkpoint)
    policy.to(args.device)
    policy.eval()
    print("Policy loaded successfully!")
    
    # Load action normalization stats for unnormalization
    print("\nLoading action normalization stats...")
    action_mean, action_std = load_action_stats(args.checkpoint)
    if action_mean is not None:
        print(f"  action_mean: {action_mean}")
        print(f"  action_std: {action_std}")
    else:
        print("  Warning: Could not load action stats, actions will not be unnormalized!")
    
    # Create environment
    print(f"\nCreating environment {args.env_id}...")
    env = gym.make(
        args.env_id,
        obs_mode="sensor_data",
        control_mode="pd_joint_pos",
        render_mode="rgb_array" if args.save_video else None,
        num_envs=1,
    )
    print("Environment created successfully!")
    
    # Create video directory if needed
    if args.save_video:
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluation loop
    successes = []
    rewards_all = []
    
    for episode in tqdm(range(args.num_episodes), desc="Evaluating"):
        obs, info = env.reset(seed=args.seed + episode)
        
        # Reset policy state (for action chunking)
        policy.reset()
        
        episode_reward = 0
        episode_success = False
        frames = []
        
        for step in range(args.max_steps):
            # Convert observation to policy input format
            debug_this_step = args.debug and step == 0 and episode == 0
            policy_obs = get_observation_for_policy(obs, args.device, debug=debug_this_step)
            
            # Get action from policy
            # Note: select_action() returns normalized action, we need to unnormalize it
            with torch.no_grad():
                action = policy.select_action(policy_obs)
            
            # Convert action to numpy for environment
            if isinstance(action, torch.Tensor):
                action = action.cpu().numpy()
            
            # Ensure action has correct shape
            if action.ndim == 1:
                action = action.reshape(1, -1)
            
            # Unnormalize action: unnorm = norm * std + mean
            if action_mean is not None and action_std is not None:
                action = action * action_std + action_mean
            
            # Clip action to environment's action space
            action = np.clip(action, env.action_space.low, env.action_space.high)
            
            if debug_this_step:
                print(f"\n[DEBUG] Action stats (after unnormalization):")
                print(f"  action: shape={action.shape}, values={action}")
                print(f"  action min={action.min():.3f}, max={action.max():.3f}")
                # Also print the environment's action space
                print(f"\n[DEBUG] Environment action space:")
                print(f"  action_space: {env.action_space}")
                if hasattr(env.action_space, 'low'):
                    print(f"  action_space.low: {env.action_space.low}")
                    print(f"  action_space.high: {env.action_space.high}")
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward.item() if hasattr(reward, 'item') else reward
            
            # Collect frames for video
            if args.save_video:
                frame = env.render()
                if frame is not None:
                    if isinstance(frame, torch.Tensor):
                        frame = frame.cpu().numpy()
                    if frame.ndim == 4:
                        frame = frame[0]  # Take first env
                    frames.append(frame)
            
            # Check if done
            done = terminated or truncated
            if hasattr(done, 'item'):
                done = done.item()
            elif hasattr(done, '__getitem__'):
                done = done[0]
            
            if done:
                break
        
        # Check success
        success = info.get('success', False)
        if hasattr(success, 'item'):
            success = success.item()
        elif hasattr(success, '__getitem__'):
            success = success[0]
        
        successes.append(success)
        rewards_all.append(episode_reward)
        episode_success = success
        
        # Save video for this episode
        if args.save_video and frames:
            video_path = video_dir / f"episode_{episode:03d}_{'success' if episode_success else 'fail'}.mp4"
            # Process frames
            processed_frames = []
            for frame in frames:
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = (frame * 255).astype(np.uint8)
                    else:
                        frame = frame.astype(np.uint8)
                processed_frames.append(frame)
            imageio.mimsave(str(video_path), processed_frames, fps=20)
        
        # Print progress every 10 episodes
        if (episode + 1) % 10 == 0:
            current_success_rate = np.mean(successes) * 100
            print(f"\n  Episodes {episode + 1}/{args.num_episodes}: "
                  f"Success Rate = {current_success_rate:.1f}%")
    
    env.close()
    
    # Print final results
    success_rate = np.mean(successes) * 100
    avg_reward = np.mean(rewards_all)
    
    print(f"\n" + "=" * 60)
    print(f"EVALUATION RESULTS")
    print(f"=" * 60)
    print(f"Environment: {args.env_id}")
    print(f"Total Episodes: {args.num_episodes}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Reward: {avg_reward:.4f}")
    print(f"Successes: {sum(successes)}/{args.num_episodes}")
    print(f"=" * 60)
    
    if args.save_video:
        print(f"\nVideos saved to: {args.video_dir}/")


if __name__ == "__main__":
    main()

