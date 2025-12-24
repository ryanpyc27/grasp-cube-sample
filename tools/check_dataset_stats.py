#!/usr/bin/env python3
"""
Check dataset statistics to debug action/state alignment issues.
"""

import argparse
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--root", type=str, required=True)
    args = parser.parse_args()
    
    print(f"Loading dataset...")
    dataset = LeRobotDataset(repo_id=args.repo_id, root=args.root)
    
    print(f"\nDataset info:")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Num episodes: {dataset.num_episodes}")
    
    # Collect statistics
    all_states = []
    all_actions = []
    
    print(f"\nCollecting statistics...")
    for i in range(min(len(dataset), 1000)):  # Sample first 1000 frames
        item = dataset[i]
        if 'observation.state' in item:
            state = item['observation.state']
            if hasattr(state, 'numpy'):
                state = state.numpy()
            all_states.append(state)
        if 'action' in item:
            action = item['action']
            if hasattr(action, 'numpy'):
                action = action.numpy()
            all_actions.append(action)
    
    all_states = np.array(all_states)
    all_actions = np.array(all_actions)
    
    print(f"\n" + "=" * 60)
    print("STATE STATISTICS")
    print("=" * 60)
    print(f"Shape: {all_states.shape}")
    print(f"\nPer-dimension statistics:")
    for i in range(all_states.shape[1]):
        print(f"  state[{i}]: min={all_states[:, i].min():.4f}, "
              f"max={all_states[:, i].max():.4f}, "
              f"mean={all_states[:, i].mean():.4f}, "
              f"std={all_states[:, i].std():.4f}")
    
    print(f"\n" + "=" * 60)
    print("ACTION STATISTICS")
    print("=" * 60)
    print(f"Shape: {all_actions.shape}")
    print(f"\nPer-dimension statistics:")
    for i in range(all_actions.shape[1]):
        print(f"  action[{i}]: min={all_actions[:, i].min():.4f}, "
              f"max={all_actions[:, i].max():.4f}, "
              f"mean={all_actions[:, i].mean():.4f}, "
              f"std={all_actions[:, i].std():.4f}")
    
    # Print first few samples
    print(f"\n" + "=" * 60)
    print("FIRST FEW SAMPLES")
    print("=" * 60)
    for i in range(min(5, len(dataset))):
        item = dataset[i]
        print(f"\nFrame {i}:")
        if 'observation.state' in item:
            state = item['observation.state']
            if hasattr(state, 'numpy'):
                state = state.numpy()
            print(f"  state: {state}")
        if 'action' in item:
            action = item['action']
            if hasattr(action, 'numpy'):
                action = action.numpy()
            print(f"  action: {action}")


if __name__ == "__main__":
    main()

