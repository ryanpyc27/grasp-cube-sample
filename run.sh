CUDA_VISIBLE_DEVICES=0 python -m grasp_cube.motionplanning.so101.run \
    -n 200 \
    -e StackCubeSO101-v1 \
    --obs-mode sensor_data \
    --only-count-success \
    --save-video

CUDA_VISIBLE_DEVICES=1 python -m grasp_cube.motionplanning.so101.run \
    -n 200 \
    -e SortCubeSO101-v1 \
    --obs-mode sensor_data \
    --only-count-success

python convert_stack_cube_to_lerobot.py \
    --h5-path /dataset/grasp-cube/demos/StackCubeSO101-v1/motionplanning/20251223_054812.h5

HF_HUB_OFFLINE=1 lerobot-train \
    --dataset.root=/dataset/grasp-cube/lerobot/stack_cube-StackCubeSO101-v1-pd_joint_pos-sensor_data-default \
    --dataset.repo_id=stack_cube \
    --policy.type=act \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=sort_cubes_act \
    --output_dir=/homes/yichengp/grasp-cube-sample/log/stack_cube_200samples \
    --dataset.revision=main

rm -r /root/autodl-tmp/lerobot_outputs

python train.py env_id=StackCubeSO101-v1

CUDA_VISIBLE_DEVICES=2 python eval_policy.py \
    --checkpoint /homes/yichengp/grasp-cube-sample/log/stack_cube_200samples/checkpoints/last/pretrained_model \
    --env-id StackCubeSO101-v1 \
    --num-episodes 10 \
    --max-steps 200 \
    --save-video \
    --debug