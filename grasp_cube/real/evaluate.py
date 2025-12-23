import dataclasses
from lerobot_env import LeRobotEnv, LeRobotEnvConfig
from .act_policy import LeRobotACTPolicy, LeRobotACTPolicyConfig
from collections import deque
import tyro

@dataclasses.dataclass
class Args:
    policy: LeRobotACTPolicyConfig
    env: LeRobotEnvConfig
    replan_steps: int | None = 32
    num_episodes: int = 10
    
def main(args: Args):
    env = LeRobotEnv(args.env)
    policy = LeRobotACTPolicy(args.policy)
    
    for episode in range(args.num_episodes):
        obs, info = env.reset()
        policy.reset()
        done = False
        step = 0
        action_queue = deque()
        while not done:
            if not action_queue:
                action = policy.get_action(obs)
                # assert action shape
                assert action.ndim == 2, f"Expected action to have 2 dimensions, got {action.ndim}"
                if args.replan_steps is not None:
                    assert action.shape[0] >= args.replan_steps, f"Expected action shape[0] <= {args.replan_steps}, got {action.shape[0]}"
                    action = action[:args.replan_steps]
                assert action.shape[1] == env.action_dim
                action_queue.extend(action)
            action = action_queue.popleft()
            obs, reward, done, truncated, info = env.step(action)
            step += 1
        print(f"Episode {episode+1} finished.")
    
    env.close()
    
if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
    
"""
python ./evaluate.py \
    --env.no-dry-run \
    --env.camera-config-path configs/bi_so101.json \
    policy:le-robot-diffusion-policy-config \
    --policy.path ~/Desktop/train/diffusion_sort/checkpoints/last/pretrained_model \
    --policy.robot-type bi_so101 \
    env.robot:bi-so101-follower-config \
    --env.robot.id eai-robot-bi-follower \
    --env.robot.right-arm-port /dev/ttyACM0 \
    --env.robot.left-arm-port /dev/ttyACM2 \
    env.teleop:bi-so101-leader-config \
    --env.teleop.right-arm-port /dev/ttyACM1 \
    --env.teleop.left-arm-port /dev/ttyACM3 \
    --env.teleop.id eai-robot-bi-leader
    
python ./evaluate.py \
    --policy.path ~/Desktop/train/act_lift/checkpoints/last/pretrained_model \
    --policy.robot-type so101 \
    --env.no-dry-run \
    --env.camera-config-path configs/so101.json \
    env.robot:so101-follower-config \
    --env.robot.id eai-robot-right-follower-arm \
    --env.robot.port /dev/ttyACM0 \
    env.teleop:so101-leader-config \
    --env.teleop.port /dev/ttyACM1 \
    --env.teleop.id eai-robot-right-leader-arm
"""