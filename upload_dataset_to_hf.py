#!/usr/bin/env python3
"""
Upload a local LeRobot dataset to Hugging Face Hub.

Usage:
    python upload_dataset_to_hf.py \
        --repo-id stack_cube \
        --root /dataset/grasp-cube/lerobot/stack_cube-StackCubeSO101-v1-pd_joint_pos-sensor_data-default \
        --hf-repo-id RyanPan315464/stack_cube_so101
"""

import argparse
from huggingface_hub import HfApi, create_repo
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Upload LeRobot dataset to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Local dataset repo ID"
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Path to local dataset root directory"
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        required=True,
        help="Hugging Face repo ID (e.g., YOUR_USERNAME/dataset_name)"
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Make the dataset private on Hugging Face"
    )
    parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=["robotics", "manipulation", "so101"],
        help="Tags for the dataset"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # First, create the repo on Hugging Face if it doesn't exist
    print(f"\nCreating repo on Hugging Face: {args.hf_repo_id}")
    try:
        create_repo(
            repo_id=args.hf_repo_id,
            repo_type="dataset",
            private=args.private,
            exist_ok=True,  # Don't error if already exists
        )
        print(f"  ✓ Repo created/verified: {args.hf_repo_id}")
    except Exception as e:
        print(f"  Warning: {e}")
    
    print(f"\nLoading dataset...")
    print(f"  Local repo_id: {args.repo_id}")
    print(f"  Root: {args.root}")
    
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.root,
    )
    
    print(f"\nDataset info:")
    print(f"  Total frames: {len(dataset)}")
    print(f"  Number of episodes: {dataset.num_episodes}")
    
    print(f"\nUploading to Hugging Face Hub...")
    print(f"  HF repo: {args.hf_repo_id}")
    print(f"  Private: {args.private}")
    print(f"  Tags: {args.tags}")
    
    # Update the dataset's repo_id to match the target
    dataset.repo_id = args.hf_repo_id
    
    dataset.push_to_hub(
        tags=args.tags,
        private=args.private,
        push_videos=True,
        license="apache-2.0",
    )
    
    print(f"\n✓ Successfully uploaded!")
    print(f"  View at: https://huggingface.co/datasets/{args.hf_repo_id}")


if __name__ == "__main__":
    main()

