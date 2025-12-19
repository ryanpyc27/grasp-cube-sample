#!/usr/bin/env python3
"""
Quick test script for the dual-arm SortCubeSO101 environment
"""
import gymnasium as gym
import mani_skill.envs
import grasp_cube.envs.tasks.sort_cube_so101
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper

def main():
    print("Creating SortCubeSO101-v1 environment...")
    env = gym.make(
        "SortCubeSO101-v1",
        num_envs=1,
        obs_mode="state_dict",
        control_mode="pd_joint_delta_pos",
        render_mode="rgb_array",
    )
    
    # For multi-agent environments, flatten the Dict action space
    print("Flattening action space for multi-agent environment...")
    env = FlattenActionSpaceWrapper(env)
    
    print("Resetting environment...")
    obs, info = env.reset(seed=0)
    
    print("\n=== Environment Info ===")
    print(f"Number of robots: {len(env.unwrapped.agent.agents)}")
    print(f"Robot 1 position: {env.unwrapped.robot1_position}")
    print(f"Robot 2 position: {env.unwrapped.robot2_position}")
    print(f"Red cube target: {env.unwrapped.red_cube_target_region_center}")
    print(f"Green cube target: {env.unwrapped.green_cube_target_region_center}")
    
    print("\n=== Observation Keys ===")
    for key in obs.keys():
        if isinstance(obs[key], dict):
            print(f"{key}:")
            for subkey in obs[key].keys():
                print(f"  - {subkey}: {obs[key][subkey].shape if hasattr(obs[key][subkey], 'shape') else type(obs[key][subkey])}")
        else:
            print(f"{key}: {obs[key].shape if hasattr(obs[key], 'shape') else type(obs[key])}")
    
    print("\n=== Action Space ===")
    print(f"Action space: {env.action_space}")
    print(f"Action space shape: {env.action_space.shape}")
    
    print("\n=== Testing step ===")
    action = env.action_space.sample()
    print(f"Sample action shape: {action.shape}")
    obs, reward, terminated, truncated, info = env.step(action)
    print(f"Step successful!")
    print(f"Reward: {reward}")
    print(f"Info keys: {list(info.keys())}")
    
    print("\n=== Success! ===")
    print("Environment created and tested successfully!")
    
    env.close()

if __name__ == "__main__":
    main()










