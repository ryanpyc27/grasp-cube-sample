python eval_policy.py \
    --policy.path log/self_defined_100ksteps/checkpoints/040000/pretrained_model \
    --policy.robot-type bi_so101 \
    --policy.act-steps 16 \
    --policy.device cuda:0 \
    --env-id SelfDefinedSO101-v1 \
    --num-episodes 100

python eval_policy.py \
    --policy.path log/self_defined_100ksteps_custom/checkpoints/040000/pretrained_model \
    --policy.robot-type bi_so101 \
    --policy.act-steps 16 \
    --policy.device cuda:0 \
    --env-id SelfDefinedSO101-v1 \
    --num-episodes 100 \
    --policy-type custom_act