#!/usr/bin/env python3
"""
Test script for SelfDefinedSO101 environment - renders and saves images
"""
import gymnasium as gym
import numpy as np
import mani_skill.envs
import grasp_cube.envs.tasks.self_defined_so101
from mani_skill.utils.wrappers import FlattenActionSpaceWrapper
from PIL import Image
from pathlib import Path
import tyro
from dataclasses import dataclass

@dataclass
class Args:
    output_dir: str = "output_images"
    num_steps: int = 10
    seed: int = 0
    shader: str = "default"
    save_sensor_images: bool = True

def main(args: Args):
    print("Creating SelfDefinedSO101-v1 environment...")
    
    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    print(f"Saving images to: {output_path}")
    
    # Create environment with rgb_array render mode for saving images
    env = gym.make(
        "SelfDefinedSO101-v1",
        num_envs=1,
        obs_mode="sensor_data",  # Use sensor_data to get camera images
        control_mode="pd_joint_pos",
        render_mode="rgb_array",  # This allows us to get rendered images
        sensor_configs=dict(shader_pack=args.shader),
        human_render_camera_configs=dict(shader_pack=args.shader),
    )
    
    # For multi-agent environments, flatten the Dict action space
    print("Flattening action space for multi-agent environment...")
    env = FlattenActionSpaceWrapper(env)
    
    print("Resetting environment...")
    obs, info = env.reset(seed=args.seed)
    
    print("\n=== Environment Info ===")
    print(f"Number of robots: {len(env.unwrapped.agent.agents)}")
    print(f"Robot 1 position: {env.unwrapped.robot1_position}")
    print(f"Robot 2 position: {env.unwrapped.robot2_position}")
    
    print("\n=== Observation Keys ===")
    for key in obs.keys():
        if isinstance(obs[key], dict):
            print(f"{key}:")
            for subkey in obs[key].keys():
                if hasattr(obs[key][subkey], 'shape'):
                    print(f"  - {subkey}: {obs[key][subkey].shape}")
                else:
                    print(f"  - {subkey}: {type(obs[key][subkey])}")
        else:
            if hasattr(obs[key], 'shape'):
                print(f"{key}: {obs[key].shape}")
            else:
                print(f"{key}: {type(obs[key])}")
    
    # Save initial observation images
    if args.save_sensor_images and 'sensor_data' in obs:
        print("\n=== Saving sensor images ===")
        sensor_data = obs['sensor_data']
        
        # Save base camera
        if 'base_camera' in sensor_data and 'Color' in sensor_data['base_camera']:
            base_img = sensor_data['base_camera']['Color'][0]  # [0] for first env
            # Convert to numpy if it's a tensor
            if hasattr(base_img, 'cpu'):
                base_img = base_img.cpu().numpy()
            # Convert from float [0,1] to uint8 [0,255]
            if base_img.dtype == np.float32 or base_img.dtype == np.float64:
                base_img = (np.clip(base_img, 0, 1) * 255).astype(np.uint8)
            # If RGBA, convert to RGB
            if base_img.shape[-1] == 4:
                base_img = base_img[..., :3]
            Image.fromarray(base_img).save(output_path / f"step_0_base_camera.png")
            print(f"  Saved: step_0_base_camera.png")
        
        # Save wrist camera 1
        if 'wrist_camera_1' in sensor_data and 'Color' in sensor_data['wrist_camera_1']:
            wrist1_img = sensor_data['wrist_camera_1']['Color'][0]
            if hasattr(wrist1_img, 'cpu'):
                wrist1_img = wrist1_img.cpu().numpy()
            if wrist1_img.dtype == np.float32 or wrist1_img.dtype == np.float64:
                wrist1_img = (np.clip(wrist1_img, 0, 1) * 255).astype(np.uint8)
            if wrist1_img.shape[-1] == 4:
                wrist1_img = wrist1_img[..., :3]
            Image.fromarray(wrist1_img).save(output_path / f"step_0_wrist_camera_1.png")
            print(f"  Saved: step_0_wrist_camera_1.png")
        
        # Save wrist camera 2
        if 'wrist_camera_2' in sensor_data and 'Color' in sensor_data['wrist_camera_2']:
            wrist2_img = sensor_data['wrist_camera_2']['Color'][0]
            if hasattr(wrist2_img, 'cpu'):
                wrist2_img = wrist2_img.cpu().numpy()
            if wrist2_img.dtype == np.float32 or wrist2_img.dtype == np.float64:
                wrist2_img = (np.clip(wrist2_img, 0, 1) * 255).astype(np.uint8)
            if wrist2_img.shape[-1] == 4:
                wrist2_img = wrist2_img[..., :3]
            Image.fromarray(wrist2_img).save(output_path / f"step_0_wrist_camera_2.png")
            print(f"  Saved: step_0_wrist_camera_2.png")
    
    # Render from human camera and save
    print("\n=== Rendering human camera view ===")
    render_img = env.render()
    if render_img is not None:
        # Convert to numpy if it's a tensor
        if hasattr(render_img, 'cpu'):
            render_img = render_img.cpu().numpy()
        # render_img should be numpy array (H, W, 3) uint8
        if len(render_img.shape) == 4:  # Batched: (B, H, W, 3)
            render_img = render_img[0]
        Image.fromarray(render_img).save(output_path / f"step_0_render.png")
        print(f"  Saved: step_0_render.png")
    
    # Run a few steps
    print(f"\n=== Running {args.num_steps} steps ===")
    for step in range(1, args.num_steps + 1):
        # Sample random action
        action = env.action_space.sample() * 0  # Zero action to keep robots still
        obs, reward, terminated, truncated, info = env.step(action)
        
        if step % 5 == 0 or step == 1:
            print(f"Step {step}: reward={reward[0]:.4f}")
            
            # Save sensor images
            if args.save_sensor_images and 'sensor_data' in obs:
                sensor_data = obs['sensor_data']
                
                if 'base_camera' in sensor_data and 'Color' in sensor_data['base_camera']:
                    base_img = sensor_data['base_camera']['Color'][0]
                    if hasattr(base_img, 'cpu'):
                        base_img = base_img.cpu().numpy()
                    if base_img.dtype == np.float32 or base_img.dtype == np.float64:
                        base_img = (np.clip(base_img, 0, 1) * 255).astype(np.uint8)
                    if base_img.shape[-1] == 4:
                        base_img = base_img[..., :3]
                    Image.fromarray(base_img).save(output_path / f"step_{step}_base_camera.png")
            
            # Save render image
            render_img = env.render()
            if render_img is not None:
                if hasattr(render_img, 'cpu'):
                    render_img = render_img.cpu().numpy()
                if len(render_img.shape) == 4:
                    render_img = render_img[0]
                Image.fromarray(render_img).save(output_path / f"step_{step}_render.png")
        
        if terminated[0] or truncated[0]:
            print(f"Episode ended at step {step}")
            break
    
    print("\n=== Success! ===")
    print(f"Images saved to: {output_path.absolute()}")
    
    env.close()

if __name__ == "__main__":
    main(tyro.cli(Args))

