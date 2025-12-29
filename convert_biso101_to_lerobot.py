import shutil
import json
from pathlib import Path

import gymnasium as gym
import numpy as np

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import h5py
import tyro
import math
import mani_skill
import grasp_cube.envs.tasks.sort_cube_so101
import grasp_cube.envs.tasks.self_defined_so101

REPO_NAME = "self_defined"

def main(h5_path: Path, *, push_to_hub: bool = False):
    json_path = h5_path.with_suffix('.json')
    assert json_path.exists(), f"JSON file not found: {json_path}"
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    assert 'env_info' in json_data, "JSON file does not contain 'env_info'"
    
    h5_file = h5py.File(h5_path, 'r')
    
    # SortCubeSO101 (BiSO101) uses 'Color' (RGBA) instead of 'rgb'
    # Shape is (T, H, W, 4) for RGBA, we'll convert to RGB (H, W, 3)
    # BiSO101 has dual arms with two wrist cameras
    base_camera_color = h5_file['traj_0']['obs']['sensor_data']['base_camera']['Color']
    wrist_camera_1_color = h5_file['traj_0']['obs']['sensor_data']['wrist_camera_1']['Color']
    wrist_camera_2_color = h5_file['traj_0']['obs']['sensor_data']['wrist_camera_2']['Color']
    
    # Convert RGBA float32 to RGB uint8
    image_shape = (base_camera_color.shape[1], base_camera_color.shape[2], 3)  # (H, W, 3)
    wrist_image_shape = (wrist_camera_1_color.shape[1], wrist_camera_1_color.shape[2], 3)
    
    # BiSO101 has two agents (so101-0 and so101-1), each with 6D qpos (4 joints + 2 gripper)
    agent_0_qpos = h5_file['traj_0']['obs']['agent']['so101-0']['qpos']
    agent_1_qpos = h5_file['traj_0']['obs']['agent']['so101-1']['qpos']
    state_shape = (agent_0_qpos.shape[1] + agent_1_qpos.shape[1],)  # Concatenate both arms
    action_shape = h5_file['traj_0']['actions'].shape[1:]
    
    control_mode = json_data['env_info']['env_kwargs']['control_mode']
    obs_mode = json_data['env_info']['env_kwargs']['obs_mode']
    shader_type = json_data['env_info']['env_kwargs']['sensor_configs']['shader_pack']
    env_id = json_data['env_info']['env_id']
    
    env = gym.make(
        env_id,
        **json_data['env_info']['env_kwargs'],
    )
    
    repo_name = f"{REPO_NAME}-{env_id}-{control_mode}-{obs_mode}-{shader_type}"

    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        confirm = input(f"Output path {output_path} already exists. Do you want to delete it? (y/n): ")
        if confirm.lower() == 'y':
            print(f"Deleting {output_path}...")
            shutil.rmtree(output_path)
        elif confirm.lower() == 'n':
            print("Exiting without deleting the existing dataset.")
            return

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="biso101",  # Bi-arm SO101
        fps=env.unwrapped._control_freq,
        features={
            "observation.images.image": {
                "dtype": "image",
                "shape": image_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist_image_1": {
                "dtype": "image",
                "shape": wrist_image_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.images.wrist_image_2": {
                "dtype": "image",
                "shape": wrist_image_shape,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": state_shape,
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": action_shape,
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    
    print(f"Converting {len(h5_file.keys())} trajectories...")
    
    for traj_idx in range(len(h5_file.keys())):
        if traj_idx % 10 == 0:
            print(f"Processing trajectory {traj_idx}/{len(h5_file.keys())}...")
            
        traj = h5_file[f'traj_{traj_idx}']
        obs = traj['obs']
        actions = traj['actions']
        
        for step in range(actions.shape[0]):
            # Convert RGBA float32 [0,1] to RGB uint8 [0,255]
            base_color = obs['sensor_data']['base_camera']['Color'][step]
            wrist_1_color = obs['sensor_data']['wrist_camera_1']['Color'][step]
            wrist_2_color = obs['sensor_data']['wrist_camera_2']['Color'][step]
            
            # Take only RGB channels (first 3), convert to uint8
            image = (np.clip(base_color[..., :3], 0, 1) * 255).astype(np.uint8)
            wrist_image_1 = (np.clip(wrist_1_color[..., :3], 0, 1) * 255).astype(np.uint8)
            wrist_image_2 = (np.clip(wrist_2_color[..., :3], 0, 1) * 255).astype(np.uint8)
            
            # Get qpos for both arms
            agent_0_qpos_step = obs['agent']['so101-0']['qpos'][step]
            agent_1_qpos_step = obs['agent']['so101-1']['qpos'][step]
            
            action = actions[step]
            
            # State: concatenate both arms' states (each arm: 4 joints + 2 gripper = 6D)
            state = np.concatenate([agent_0_qpos_step, agent_1_qpos_step], axis=-1)
            
            dataset.add_frame({
                "observation.images.image": image,
                "observation.images.wrist_image_1": wrist_image_1,
                "observation.images.wrist_image_2": wrist_image_2,
                "observation.state": state,
                "action": action,
                "task": env_id,
            })      
            
        dataset.save_episode()
    
    print(f"\nConversion complete! Converted {len(h5_file.keys())} episodes.")
    print(f"Dataset saved to: {output_path}")
    
    if push_to_hub:
        print("\nPushing to HuggingFace Hub...")
        dataset.push_to_hub(
            tags=["self_defined", "biso101"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Successfully pushed to hub!")
        

if __name__ == "__main__":
    tyro.cli(main)

