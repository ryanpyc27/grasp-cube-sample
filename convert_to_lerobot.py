"""
Convert ManiSkill HDF5 trajectory to LeRobot Dataset format.

LeRobot Dataset v2.0 format:
- data/chunk-XXX/episode_XXXXXX.parquet  (observation & action data)
- videos/observation.images.XXX/episode_XXXXXX.mp4  (video files)
- meta/info.json  (dataset configuration)
- meta/stats.json  (statistics)
- meta/episodes.jsonl  (episode metadata)
- meta/tasks.jsonl  (task descriptions)

Usage:
    python convert_to_lerobot.py --input demos/SortCubeSO101-v1/motionplanning/20251221_120352.h5 \
                                  --output lerobot_dataset/sort_cubes
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Any

import h5py
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args():
    parser = argparse.ArgumentParser(description="Convert ManiSkill HDF5 to LeRobot Dataset format")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input HDF5 file")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to output LeRobot dataset directory")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    parser.add_argument("--repo-id", type=str, default="local/sort_cubes", help="Dataset repository ID")
    parser.add_argument("--robot-type", type=str, default="so101_dual_arm", help="Robot type")
    parser.add_argument("--task", type=str, default="Sort cubes by color", help="Task description")
    parser.add_argument("--video-backend", type=str, default="pyav", help="Video backend")
    return parser.parse_args()


def extract_h5_data(h5_path: str) -> Dict[str, Any]:
    """Extract all trajectory data from HDF5 file."""
    episodes = []
    
    with h5py.File(h5_path, 'r') as f:
        traj_keys = sorted(f.keys(), key=lambda x: int(x.split('_')[1]))
        
        for traj_key in traj_keys:
            traj = f[traj_key]
            episode_data = {
                'traj_key': traj_key,
                'actions': traj['actions'][:],
                'terminated': traj['terminated'][:],
                'truncated': traj['truncated'][:],
            }
            
            # Extract success if available
            if 'success' in traj:
                episode_data['success'] = traj['success'][:]
            
            # Extract observations
            if 'obs' in traj:
                obs_group = traj['obs']
                episode_data['obs'] = {}
                
                # Agent state (qpos, qvel)
                if 'agent' in obs_group:
                    episode_data['obs']['agent'] = {}
                    for agent_name in obs_group['agent'].keys():
                        agent_data = obs_group['agent'][agent_name]
                        episode_data['obs']['agent'][agent_name] = {
                            'qpos': agent_data['qpos'][:] if 'qpos' in agent_data else None,
                            'qvel': agent_data['qvel'][:] if 'qvel' in agent_data else None,
                        }
                
                # Sensor data (RGB images)
                if 'sensor_data' in obs_group:
                    episode_data['obs']['sensor_data'] = {}
                    for sensor_name in obs_group['sensor_data'].keys():
                        sensor = obs_group['sensor_data'][sensor_name]
                        if 'rgb' in sensor:
                            episode_data['obs']['sensor_data'][sensor_name] = {
                                'rgb': sensor['rgb'][:]
                            }
                
                # Extra info (target positions, tcp pose, etc.)
                if 'extra' in obs_group:
                    episode_data['obs']['extra'] = {}
                    for key in obs_group['extra'].keys():
                        episode_data['obs']['extra'][key] = obs_group['extra'][key][:]
            
            # Extract environment states
            if 'env_states' in traj:
                episode_data['env_states'] = {}
                env_states = traj['env_states']
                if 'articulations' in env_states:
                    episode_data['env_states']['articulations'] = {}
                    for art_name in env_states['articulations'].keys():
                        episode_data['env_states']['articulations'][art_name] = \
                            env_states['articulations'][art_name][:]
            
            episodes.append(episode_data)
    
    return episodes


def create_episode_parquet(episode_data: Dict, episode_idx: int, fps: int) -> pa.Table:
    """Create a parquet table for a single episode."""
    num_frames = len(episode_data['actions'])
    
    # Build columns
    columns = {
        'frame_index': list(range(num_frames)),
        'episode_index': [episode_idx] * num_frames,
        'index': list(range(num_frames)),  # Global index, will be updated later
        'task_index': [0] * num_frames,
        'timestamp': [i / fps for i in range(num_frames)],
    }
    
    # Add actions (flatten to 1D array per frame)
    actions = episode_data['actions']
    columns['action'] = [actions[i].tolist() for i in range(num_frames)]
    
    # Add state observations (qpos from both arms)
    if 'obs' in episode_data and 'agent' in episode_data['obs']:
        state_list = []
        for i in range(num_frames):
            state = []
            for agent_name in sorted(episode_data['obs']['agent'].keys()):
                agent_obs = episode_data['obs']['agent'][agent_name]
                if agent_obs['qpos'] is not None:
                    # Use frame i+1 for state since obs has T+1 frames
                    frame_idx = min(i + 1, len(agent_obs['qpos']) - 1)
                    state.extend(agent_obs['qpos'][frame_idx].tolist())
            state_list.append(state)
        columns['observation.state'] = state_list
    
    # Add extra observations (tcp poses, target positions)
    if 'obs' in episode_data and 'extra' in episode_data['obs']:
        extra_list = []
        for i in range(num_frames):
            extra = []
            for key in sorted(episode_data['obs']['extra'].keys()):
                frame_idx = min(i + 1, len(episode_data['obs']['extra'][key]) - 1)
                extra.extend(episode_data['obs']['extra'][key][frame_idx].tolist())
            extra_list.append(extra)
        columns['observation.environment_state'] = extra_list
    
    # NOTE: Video data is stored in separate mp4 files, not in parquet
    # LeRobot v3.0 loads videos separately based on episode_index and frame_index
    
    # Add next done flag
    done = episode_data['terminated'] | episode_data['truncated']
    columns['next.done'] = done.tolist()
    
    # Add next reward (placeholder, can be customized)
    if 'success' in episode_data:
        # Use success as sparse reward
        columns['next.reward'] = episode_data['success'].astype(float).tolist()
    else:
        columns['next.reward'] = [0.0] * num_frames
    
    # Add next success
    if 'success' in episode_data:
        columns['next.success'] = episode_data['success'].tolist()
    
    return pa.table(columns)


def save_episode_video(episode_data: Dict, output_dir: Path, episode_idx: int, fps: int):
    """Save episode images as video using imageio."""
    try:
        import imageio
    except ImportError:
        print("Warning: imageio not installed, skipping video generation")
        print("Install with: pip install imageio[ffmpeg]")
        return
    
    if 'obs' not in episode_data or 'sensor_data' not in episode_data['obs']:
        return
    
    for sensor_name, sensor_data in episode_data['obs']['sensor_data'].items():
        if 'rgb' not in sensor_data:
            continue
        
        rgb_images = sensor_data['rgb']
        video_dir = output_dir / "videos" / f"observation.images.{sensor_name}"
        video_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = video_dir / f"episode_{episode_idx:06d}.mp4"
        
        # Use actions length for video frames (skip the first observation frame)
        num_action_frames = len(episode_data['actions'])
        
        # Write video
        writer = imageio.get_writer(str(video_path), fps=fps, codec='libx264', 
                                    output_params=['-crf', '23'])
        for i in range(num_action_frames):
            # Use frame i+1 from observations (to match action timing)
            frame_idx = min(i + 1, len(rgb_images) - 1)
            writer.append_data(rgb_images[frame_idx])
        writer.close()
        
        print(f"  Saved video: {video_path}")


def create_meta_files(episodes: List[Dict], output_dir: Path, args):
    """Create metadata files for LeRobot dataset."""
    meta_dir = output_dir / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    
    # Gather statistics
    total_frames = sum(len(ep['actions']) for ep in episodes)
    total_episodes = len(episodes)
    
    # Get feature dimensions from first episode
    first_ep = episodes[0]
    action_dim = first_ep['actions'].shape[1]
    
    state_dim = 0
    if 'obs' in first_ep and 'agent' in first_ep['obs']:
        for agent_name in first_ep['obs']['agent'].keys():
            if first_ep['obs']['agent'][agent_name]['qpos'] is not None:
                state_dim += first_ep['obs']['agent'][agent_name]['qpos'].shape[1]
    
    env_state_dim = 0
    if 'obs' in first_ep and 'extra' in first_ep['obs']:
        for key in first_ep['obs']['extra'].keys():
            env_state_dim += first_ep['obs']['extra'][key].shape[1]
    
    # Get image info
    image_shapes = {}
    if 'obs' in first_ep and 'sensor_data' in first_ep['obs']:
        for sensor_name, sensor_data in first_ep['obs']['sensor_data'].items():
            if 'rgb' in sensor_data:
                h, w, c = sensor_data['rgb'].shape[1:]
                image_shapes[sensor_name] = {'height': h, 'width': w, 'channels': c}
    
    # Create info.json
    info = {
        "codebase_version": "v3.0",
        "robot_type": args.robot_type,
        "fps": args.fps,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": 1,
        "total_videos": total_episodes * len(image_shapes),
        "total_chunks": 1,
        "chunks_size": total_episodes,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            # Base columns (scalars use shape=[1])
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            # Action
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": None
            },
            # Observation
            "observation.state": {
                "dtype": "float32", 
                "shape": [state_dim],
                "names": None
            },
        }
    }
    
    if env_state_dim > 0:
        info["features"]["observation.environment_state"] = {
            "dtype": "float32",
            "shape": [env_state_dim],
            "names": None
        }
    
    for sensor_name, shape in image_shapes.items():
        info["features"][f"observation.images.{sensor_name}"] = {
            "dtype": "video",
            "shape": [shape['height'], shape['width'], shape['channels']],
            "video_info": {
                "video.fps": args.fps,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False
            }
        }
    
    # Add next.* columns (scalars use shape=[1])
    info["features"].update({
        "next.done": {"dtype": "bool", "shape": [1]},
        "next.reward": {"dtype": "float32", "shape": [1]},
        "next.success": {"dtype": "bool", "shape": [1]}
    })
    
    with open(meta_dir / "info.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    # Create tasks.parquet (v3.0 requires parquet format)
    tasks_df = pd.DataFrame([{"task_index": 0, "task": args.task}])
    tasks_df.to_parquet(meta_dir / "tasks.parquet", index=False)
    
    # Create episodes.parquet (v3.0 requires parquet format in nested structure)
    episodes_list = []
    for i, ep in enumerate(episodes):
        episode_info = {
            "episode_index": i,
            "tasks": [args.task],
            "length": len(ep['actions'])
        }
        # Add video chunk indices for each camera
        if 'obs' in ep and 'sensor_data' in ep['obs']:
            for sensor_name in ep['obs']['sensor_data'].keys():
                # All videos are in chunk 0 (single chunk for simplicity)
                episode_info[f"videos/observation.images.{sensor_name}/chunk_index"] = 0
        
        episodes_list.append(episode_info)
    
    episodes_df = pd.DataFrame(episodes_list)
    # LeRobot v3.0 expects episodes in a nested directory structure: meta/episodes/chunk-000/
    episodes_dir = meta_dir / "episodes" / "chunk-000"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    episodes_df.to_parquet(episodes_dir / "episodes.parquet", index=False)
    
    # Create stats.json (compute statistics)
    stats = compute_statistics(episodes)
    with open(meta_dir / "stats.json", 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Created metadata files in {meta_dir}")


def compute_statistics(episodes: List[Dict]) -> Dict:
    """Compute statistics for the dataset."""
    # Collect all actions
    all_actions = np.concatenate([ep['actions'] for ep in episodes], axis=0)
    
    # Collect all states
    all_states = []
    for ep in episodes:
        if 'obs' in ep and 'agent' in ep['obs']:
            for i in range(len(ep['actions'])):
                state = []
                for agent_name in sorted(ep['obs']['agent'].keys()):
                    agent_obs = ep['obs']['agent'][agent_name]
                    if agent_obs['qpos'] is not None:
                        frame_idx = min(i + 1, len(agent_obs['qpos']) - 1)
                        state.extend(agent_obs['qpos'][frame_idx].tolist())
                all_states.append(state)
    all_states = np.array(all_states) if all_states else None
    
    stats = {
        "action": {
            "min": all_actions.min(axis=0).tolist(),
            "max": all_actions.max(axis=0).tolist(),
            "mean": all_actions.mean(axis=0).tolist(),
            "std": all_actions.std(axis=0).tolist(),
        }
    }
    
    if all_states is not None and len(all_states) > 0:
        stats["observation.state"] = {
            "min": all_states.min(axis=0).tolist(),
            "max": all_states.max(axis=0).tolist(),
            "mean": all_states.mean(axis=0).tolist(),
            "std": all_states.std(axis=0).tolist(),
        }
    
    return stats


def main():
    args = parse_args()
    
    input_path = Path(args.input)
    output_dir = Path(args.output)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading HDF5 file: {input_path}")
    episodes = extract_h5_data(str(input_path))
    print(f"Found {len(episodes)} episodes")
    
    # Create data directory
    data_dir = output_dir / "data" / "chunk-000"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each episode
    global_index = 0
    for i, episode in enumerate(episodes):
        print(f"Processing episode {i+1}/{len(episodes)}...")
        
        # Create parquet file
        table = create_episode_parquet(episode, i, args.fps)
        
        # Update global index
        num_frames = len(episode['actions'])
        indices = list(range(global_index, global_index + num_frames))
        table = table.set_column(
            table.schema.get_field_index('index'),
            'index',
            pa.array(indices)
        )
        global_index += num_frames
        
        parquet_path = data_dir / f"episode_{i:06d}.parquet"
        pq.write_table(table, parquet_path)
        print(f"  Saved parquet: {parquet_path}")
        
        # Save video
        save_episode_video(episode, output_dir, i, args.fps)
    
    # Create metadata files
    create_meta_files(episodes, output_dir, args)
    
    print(f"\nConversion complete!")
    print(f"LeRobot dataset saved to: {output_dir}")
    print(f"\nTo use with LeRobot:")
    print(f"  from lerobot.common.datasets.lerobot_dataset import LeRobotDataset")
    print(f"  dataset = LeRobotDataset('{output_dir}')")


if __name__ == "__main__":
    main()

