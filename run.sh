python -m grasp_cube.motionplanning.so101.run \
    -n 200 \
    -e SortCubeSO101-v1 \
    --obs-mode rgb \
    --only-count-success \
    --lerobot-output /root/autodl-tmp/lerobot_dataset/sort_cubes

HF_HUB_OFFLINE=1 lerobot-train \
    --dataset.root=/root/autodl-tmp/lerobot_dataset/sort_cubes \
    --dataset.repo_id=sort_cubes \
    --policy.type=act \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=sort_cubes_act \
    --output_dir=/root/autodl-tmp/lerobot_outputs \
    --dataset.revision=main

rm -r /root/autodl-tmp/lerobot_outputs

python train.py env_id=StackCubeSO101-v1