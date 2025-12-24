import dataclasses
import gymnasium as gym
import numpy as np
import torch
from collections import deque
from pathlib import Path
import tyro
import imageio
from datetime import datetime

# Import custom environments
import grasp_cube.envs.tasks.stack_cube_so101
import grasp_cube.envs.tasks.sort_cube_so101

# Import LeRobot policy
from grasp_cube.real.act_policy import LeRobotACTPolicy, LeRobotACTPolicyConfig


class ManiSkillEnvWrapper:
    """Wrapper to make ManiSkill env compatible with LeRobotACTPolicy interface."""
    
    def __init__(self, env_id: str, robot_type: str = "so101", shader_pack: str = "default"):
        # Configure sensor_configs with proper shader for rendering
        sensor_configs = {
            "shader_pack": shader_pack
        }
        
        self.env = gym.make(
            env_id,
            obs_mode="sensor_data",
            control_mode="pd_joint_pos",
            render_mode="sensors",  # Enable sensor rendering
            num_envs=1,
            sensor_configs=sensor_configs,
        )
        self.robot_type = robot_type
        
    @property
    def action_dim(self) -> int:
        """Return action dimension."""
        action_space = self.env.action_space
        
        # Handle Dict action space (multi-agent environments)
        if hasattr(action_space, 'spaces'):
            # Dict space - sum up dimensions of all agents
            total_dim = 0
            for key, space in action_space.spaces.items():
                if hasattr(space, 'shape'):
                    total_dim += space.shape[-1]
            return total_dim
        # Handle Box action space (single agent)
        elif hasattr(action_space, 'shape'):
            return action_space.shape[-1]
        # Handle Discrete action space
        else:
            return action_space.n
    
    def reset(self, seed=None):
        """Reset environment and return observation in LeRobot format."""
        obs, info = self.env.reset(seed=seed)
        obs_lerobot = self._convert_obs_to_lerobot(obs)
        return obs_lerobot, info
    
    def step(self, action):
        """Step environment with action."""
        # Ensure action has correct shape [1, action_dim]
        if isinstance(action, np.ndarray):
            if action.ndim == 1:
                action = action.reshape(1, -1)
        
        # Handle Dict action space (multi-agent environments)
        action_space = self.env.action_space
        if hasattr(action_space, 'spaces'):
            # Convert flat action to Dict format
            # For bi_so101: split action into two halves for two arms
            # State input order in act_policy.py is: [right_arm, left_arm] = [so101-0, so101-1]
            # So policy output order is: [right_arm_action, left_arm_action] = [so101-0_action, so101-1_action]
            if isinstance(action, np.ndarray):
                mid = action.shape[-1] // 2
                action_dict = {
                    'so101-0': action[..., :mid],   # Right arm (so101-0) - first half
                    'so101-1': action[..., mid:]    # Left arm (so101-1) - second half
                }
                action = action_dict
        
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs_lerobot = self._convert_obs_to_lerobot(obs)
        done = terminated or truncated
        
        # Handle batched outputs
        if hasattr(done, 'item'):
            done = done.item()
        elif hasattr(done, '__getitem__'):
            done = done[0]
            
        return obs_lerobot, reward, done, truncated, info
    
    def close(self):
        """Close environment."""
        self.env.close()
    
    def _convert_obs_to_lerobot(self, obs: dict) -> dict:
        """
        Convert ManiSkill observation to LeRobot policy input format.
        
        ManiSkill provides:
        - obs['sensor_data']['base_camera']['Color'] -> RGBA float32 [B, H, W, C]
        - obs['sensor_data']['wrist_camera']['Color'] -> RGBA float32 [B, H, W, C]
        - obs['agent']['qpos'] -> state [B, state_dim]
        
        LeRobotACTPolicy expects:
        - observation["images"]["front"] -> RGB uint8 [H, W, C]
        - observation["images"]["wrist"] -> RGB uint8 [H, W, C]
        - observation["states"]["arm"] -> float32 [state_dim]
        """
        lerobot_obs = {
            "images": {},
            "states": {},
            "task": ""  # Empty task for ManiSkill
        }
        
        # Convert images from sensor_data
        if 'sensor_data' in obs:
            # Base camera (front)
            base_image = obs['sensor_data']['base_camera']['Color']
            if isinstance(base_image, torch.Tensor):
                base_image = base_image.cpu().numpy()
            # Remove batch dimension if present
            if base_image.ndim == 4:
                base_image = base_image[0]
            # Take only RGB channels (ignore Alpha)
            base_image = base_image[..., :3]
            # Convert from float [0,1] to uint8 [0,255]
            base_image = (np.clip(base_image, 0, 1) * 255).astype(np.uint8)
            lerobot_obs["images"]["front"] = base_image
            
            # Wrist camera - handle both single arm and dual arm cases
            if 'wrist_camera' in obs['sensor_data']:
                # Single arm case
                wrist_image = obs['sensor_data']['wrist_camera']['Color']
                if isinstance(wrist_image, torch.Tensor):
                    wrist_image = wrist_image.cpu().numpy()
                if wrist_image.ndim == 4:
                    wrist_image = wrist_image[0]
                wrist_image = wrist_image[..., :3]
                wrist_image = (np.clip(wrist_image, 0, 1) * 255).astype(np.uint8)
                lerobot_obs["images"]["wrist"] = wrist_image
            elif 'wrist_camera_1' in obs['sensor_data'] and 'wrist_camera_2' in obs['sensor_data']:
                # Dual arm case - use wrist_camera_1 and wrist_camera_2
                wrist_image_1 = obs['sensor_data']['wrist_camera_1']['Color']
                wrist_image_2 = obs['sensor_data']['wrist_camera_2']['Color']
                
                if isinstance(wrist_image_1, torch.Tensor):
                    wrist_image_1 = wrist_image_1.cpu().numpy()
                if isinstance(wrist_image_2, torch.Tensor):
                    wrist_image_2 = wrist_image_2.cpu().numpy()
                    
                if wrist_image_1.ndim == 4:
                    wrist_image_1 = wrist_image_1[0]
                if wrist_image_2.ndim == 4:
                    wrist_image_2 = wrist_image_2[0]
                    
                wrist_image_1 = wrist_image_1[..., :3]
                wrist_image_2 = wrist_image_2[..., :3]
                
                wrist_image_1 = (np.clip(wrist_image_1, 0, 1) * 255).astype(np.uint8)
                wrist_image_2 = (np.clip(wrist_image_2, 0, 1) * 255).astype(np.uint8)
                
                # Store both wrist cameras with proper names for bi_so101
                # wrist_camera_1 is on agent[0] (right arm) -> right_wrist
                # wrist_camera_2 is on agent[1] (left arm) -> left_wrist
                lerobot_obs["images"]["right_wrist"] = wrist_image_1
                lerobot_obs["images"]["left_wrist"] = wrist_image_2
        
        # Convert robot state (qpos)
        if 'agent' in obs:
            # Handle both single arm and dual arm observation structures
            if 'qpos' in obs['agent']:
                # Single arm case: obs['agent']['qpos']
                state = obs['agent']['qpos']
                if isinstance(state, torch.Tensor):
                    state = state.cpu().numpy()
                if state.ndim == 2:
                    state = state[0]
                    
                if self.robot_type == "so101":
                    lerobot_obs["states"]["arm"] = state.astype(np.float32)
                elif self.robot_type == "bi_so101":
                    # For bi-manual, split the state
                    mid = len(state) // 2
                    lerobot_obs["states"]["right_arm"] = state[:mid].astype(np.float32)
                    lerobot_obs["states"]["left_arm"] = state[mid:].astype(np.float32)
                    
            elif 'so101-0' in obs['agent'] and 'so101-1' in obs['agent']:
                # Dual arm case: obs['agent']['so101-0']['qpos'] and obs['agent']['so101-1']['qpos']
                state_0 = obs['agent']['so101-0']['qpos']
                state_1 = obs['agent']['so101-1']['qpos']
                
                if isinstance(state_0, torch.Tensor):
                    state_0 = state_0.cpu().numpy()
                if isinstance(state_1, torch.Tensor):
                    state_1 = state_1.cpu().numpy()
                    
                if state_0.ndim == 2:
                    state_0 = state_0[0]
                if state_1.ndim == 2:
                    state_1 = state_1[0]
                
                # Concatenate both arms' states
                state = np.concatenate([state_0, state_1], axis=-1)
                
                if self.robot_type == "bi_so101":
                    # so101-0 is agent[0] (right arm), so101-1 is agent[1] (left arm)
                    lerobot_obs["states"]["right_arm"] = state_0.astype(np.float32)
                    lerobot_obs["states"]["left_arm"] = state_1.astype(np.float32)
                else:
                    # Fallback: just use concatenated state
                    lerobot_obs["states"]["arm"] = state.astype(np.float32)
        
        return lerobot_obs


@dataclasses.dataclass
class Args:
    policy: LeRobotACTPolicyConfig
    env_id: str = "StackCubeSO101-v1"
    replan_steps: int | None = 32
    num_episodes: int = 10
    max_steps: int = 1000
    seed: int = 0
    save_video: bool = False
    video_dir: Path = Path("eval_videos")
    fps: int = 10
    shader_pack: str = "default"  # Shader pack for rendering (default, rt, rt_fast, etc.)


def main(args: Args):
    print(f"=" * 60)
    print(f"Evaluating Policy")
    print(f"=" * 60)
    print(f"Policy path: {args.policy.path}")
    print(f"Environment: {args.env_id}")
    print(f"Robot type: {args.policy.robot_type}")
    print(f"Episodes: {args.num_episodes}")
    print(f"Replan steps: {args.replan_steps}")
    print(f"Save video: {args.save_video}")
    if args.save_video:
        print(f"Video directory: {args.video_dir}")
    print(f"=" * 60)
    
    # Create video directory if saving videos
    if args.save_video:
        args.video_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_subdir = args.video_dir / f"{args.env_id}_{timestamp}"
        video_subdir.mkdir(parents=True, exist_ok=True)
        print(f"Videos will be saved to: {video_subdir}")
    
    # Create environment wrapper
    env = ManiSkillEnvWrapper(args.env_id, robot_type=args.policy.robot_type, shader_pack=args.shader_pack)
    
    # Create policy
    policy = LeRobotACTPolicy(args.policy)
    
    # Evaluation loop
    successes = []
    
    for episode in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed + episode)
        policy.reset()
        done = False
        step = 0
        action_queue = deque()
        
        # Video recording
        frames = []
        left_wrist_frames = []
        right_wrist_frames = []
        while not done and step < args.max_steps:
            # Save frame for video (using front camera)
            if args.save_video:
                # Get the front camera image in RGB uint8 format
                frame = obs["images"]["front"]  # Already in RGB uint8 [H, W, 3]
                left_wrist_frame = obs["images"]["left_wrist"]
                right_wrist_frame = obs["images"]["right_wrist"]
                frames.append(frame)
                left_wrist_frames.append(left_wrist_frame)
                right_wrist_frames.append(right_wrist_frame)
                
            
            # Get new action chunk when queue is empty
            if not action_queue:
                action = policy.get_action(obs)
                
                # Assert action shape as in evaluate.py
                assert action.ndim == 2, f"Expected action to have 2 dimensions, got {action.ndim}"
                
                # Apply replan_steps if specified
                if args.replan_steps is not None:
                    assert action.shape[0] >= args.replan_steps, f"Expected action shape[0] >= {args.replan_steps}, got {action.shape[0]}"
                    action = action[:args.replan_steps]
                
                # Check action dimension matches environment
                assert action.shape[1] == env.action_dim, f"Expected action shape[1] == {env.action_dim}, got {action.shape[1]}"
                
                # Add actions to queue
                action_queue.extend(action)
            action = action_queue.popleft()
            obs, reward, done, truncated, info = env.step(action)
            step += 1
        
        # Check success
        success = info.get('success', False)
        if hasattr(success, 'item'):
            success = success.item()
        elif hasattr(success, '__getitem__'):
            success = success[0]
        
        successes.append(success)
        
        # Save video
        if args.save_video and len(frames) > 0:
            video_filename = video_subdir / f"episode_{episode:03d}_{'success' if success else 'fail'}.mp4"
            imageio.mimsave(video_filename, frames, fps=args.fps)
            left_wrist_video_filename = video_subdir / f"episode_{episode:03d}_{'success' if success else 'fail'}_left_wrist.mp4"
            imageio.mimsave(left_wrist_video_filename, left_wrist_frames, fps=args.fps)
            right_wrist_video_filename = video_subdir / f"episode_{episode:03d}_{'success' if success else 'fail'}_right_wrist.mp4"
            imageio.mimsave(right_wrist_video_filename, right_wrist_frames, fps=args.fps)
            print(f"Episode {episode+1} finished. Success: {success}. Video saved: {video_filename}")
        else:
            print(f"Episode {episode+1} finished. Success: {success}")
    
    env.close()
    
    # Print final results
    success_rate = np.mean(successes) * 100
    print(f"\n" + "=" * 60)
    print(f"EVALUATION RESULTS")
    print(f"=" * 60)
    print(f"Environment: {args.env_id}")
    print(f"Total Episodes: {args.num_episodes}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Successes: {sum(successes)}/{args.num_episodes}")
    print(f"=" * 60)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
    
"""
Usage example:

python eval_policy.py \
    --policy.path log/stack_cube_200samples/checkpoints/last/pretrained_model \
    --policy.robot-type so101 \
    --policy.device cuda:0 \
    --env-id StackCubeSO101-v1 \
    --replan-steps 32 \
    --num-episodes 10

python eval_policy.py \
    --policy.path log/sort_cube_200samples/checkpoints/last/pretrained_model \
    --policy.robot-type bi_so101 \
    --policy.device cuda:0 \
    --env-id SortCubeSO101-v1 \
    --replan-steps 32 \
    --num-episodes 10 \
    --save-video
"""