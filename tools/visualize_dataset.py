#!/usr/bin/env python3
"""
Visualize LeRobot dataset by exporting episodes to video files.
Works on headless servers without GUI.

Usage:
    python visualize_dataset.py \
        --repo-id stack_cube \
        --root /dataset/grasp-cube/lerobot/stack_cube-StackCubeSO101-v1-pd_joint_pos-sensor_data-default \
        --episode-indices 0 1 2 \
        --output-dir dataset_videos
"""

import argparse
import numpy as np
import imageio
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize LeRobot dataset")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Dataset repo ID"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to dataset root directory"
    )
    parser.add_argument(
        "--episode-indices",
        type=int,
        nargs="+",
        default=[0],
        help="Episode indices to visualize (default: [0])"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="dataset_videos",
        help="Output directory for videos"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=20,
        help="Output video FPS"
    )
    parser.add_argument(
        "--show-state",
        action="store_true",
        help="Overlay state/action info on video"
    )
    return parser.parse_args()


def visualize_episode(dataset, episode_idx, output_dir, fps=20, show_state=False):
    """Export a single episode to video."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get episode data
    episode_data = dataset.filter(lambda x: x['episode_index'] == episode_idx)
    
    if len(episode_data) == 0:
        print(f"Episode {episode_idx} not found!")
        return
    
    print(f"\nProcessing episode {episode_idx} ({len(episode_data)} frames)...")
    
    # Check available image keys
    sample = dataset[0]
    image_keys = [k for k in sample.keys() if 'image' in k.lower()]
    print(f"Found image keys: {image_keys}")
    
    # Collect frames for each camera
    camera_frames = {key: [] for key in image_keys}
    states = []
    actions = []
    
    for i in tqdm(range(len(episode_data)), desc="Loading frames"):
        item = episode_data[i]
        
        for key in image_keys:
            if key in item:
                img = item[key]
                # Convert to numpy if tensor
                if hasattr(img, 'numpy'):
                    img = img.numpy()
                # Handle different formats
                if img.ndim == 3:
                    # Check if CHW format, convert to HWC
                    if img.shape[0] in [1, 3, 4] and img.shape[0] < img.shape[1]:
                        img = np.transpose(img, (1, 2, 0))
                # Normalize to uint8
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                camera_frames[key].append(img)
        
        if 'observation.state' in item:
            state = item['observation.state']
            if hasattr(state, 'numpy'):
                state = state.numpy()
            states.append(state)
        
        if 'action' in item:
            action = item['action']
            if hasattr(action, 'numpy'):
                action = action.numpy()
            actions.append(action)
    
    # Save video for each camera
    for key, frames in camera_frames.items():
        if not frames:
            continue
            
        # Clean up key name for filename
        clean_key = key.replace('.', '_').replace('/', '_')
        video_path = output_dir / f"episode_{episode_idx:03d}_{clean_key}.mp4"
        
        if show_state and states:
            # Create frames with state overlay
            annotated_frames = []
            for i, frame in enumerate(frames):
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(frame)
                ax.axis('off')
                
                # Add state info
                if i < len(states):
                    state_str = f"State: {np.array2string(states[i], precision=2, suppress_small=True)}"
                    ax.set_title(state_str, fontsize=8)
                if i < len(actions):
                    action_str = f"Action: {np.array2string(actions[i], precision=2, suppress_small=True)}"
                    ax.set_xlabel(action_str, fontsize=8)
                
                fig.tight_layout()
                fig.canvas.draw()
                
                # Convert figure to numpy array
                w, h = fig.canvas.get_width_height()
                buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
                buf = buf.reshape((h, w, 3))
                annotated_frames.append(buf)
                plt.close(fig)
            
            imageio.mimsave(str(video_path), annotated_frames, fps=fps)
        else:
            imageio.mimsave(str(video_path), frames, fps=fps)
        
        print(f"  Saved: {video_path}")
    
    # Also create a combined view if multiple cameras
    if len(camera_frames) > 1 and all(len(f) > 0 for f in camera_frames.values()):
        combined_frames = []
        num_frames = min(len(f) for f in camera_frames.values())
        
        for i in range(num_frames):
            row_images = []
            for key in sorted(camera_frames.keys()):
                img = camera_frames[key][i]
                # Resize to same height if needed
                row_images.append(img)
            
            # Stack horizontally
            combined = np.concatenate(row_images, axis=1)
            combined_frames.append(combined)
        
        combined_path = output_dir / f"episode_{episode_idx:03d}_combined.mp4"
        imageio.mimsave(str(combined_path), combined_frames, fps=fps)
        print(f"  Saved combined view: {combined_path}")
    
    # Save state/action plots
    if states:
        states = np.array(states)
        actions = np.array(actions) if actions else None
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot states
        ax = axes[0]
        for i in range(states.shape[1]):
            ax.plot(states[:, i], label=f'state_{i}')
        ax.set_xlabel('Frame')
        ax.set_ylabel('State Value')
        ax.set_title(f'Episode {episode_idx} - States')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True)
        
        # Plot actions
        if actions is not None:
            ax = axes[1]
            for i in range(actions.shape[1]):
                ax.plot(actions[:, i], label=f'action_{i}')
            ax.set_xlabel('Frame')
            ax.set_ylabel('Action Value')
            ax.set_title(f'Episode {episode_idx} - Actions')
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True)
        
        plt.tight_layout()
        plot_path = output_dir / f"episode_{episode_idx:03d}_states_actions.png"
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"  Saved state/action plot: {plot_path}")


def print_dataset_info(dataset):
    """Print dataset structure and info."""
    print("\n" + "=" * 60)
    print("DATASET INFO")
    print("=" * 60)
    
    print(f"Total frames: {len(dataset)}")
    print(f"Number of episodes: {dataset.num_episodes}")
    
    sample = dataset[0]
    print(f"\nData keys:")
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")
    
    print("=" * 60)


def main():
    args = parse_args()
    
    print(f"Loading dataset...")
    print(f"  repo_id: {args.repo_id}")
    print(f"  root: {args.root}")
    
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
    )
    
    print_dataset_info(dataset)
    
    # Visualize requested episodes
    for episode_idx in args.episode_indices:
        visualize_episode(
            dataset,
            episode_idx,
            args.output_dir,
            fps=args.fps,
            show_state=args.show_state
        )
    
    print(f"\n✓ Done! Videos saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

