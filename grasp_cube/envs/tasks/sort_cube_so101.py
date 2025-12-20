from typing import Any, Dict, Union

import numpy as np
import sapien
import torch

import mani_skill.envs.utils.randomization as randomization
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.envs.tasks.tabletop.pick_cube_cfgs import PICK_CUBE_CONFIGS
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose

from grasp_cube.agents.robots.so101.so_101 import SO101
from grasp_cube.envs.tasks.table import MyTableBuilder


@register_env("SortCubeSO101-v1", max_episode_steps=50)
class SortCubeSO101Env(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a red cube and move it to a target goal position. This is also the *baseline* task to test whether a robot with manipulation
    capabilities can be simulated and trained properly. Hence there is extra code for some robots to set them up properly in this environment as well as the table scene builder.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """

    SUPPORTED_ROBOTS = [
        ("so101", "so101"),
    ]
    cube_half_size = 0.015
    goal_thresh = 0.015 * 1.25
    cube_spawn_half_size = (0.078, 0.082)
    cube_spawn_center = (0.30, 0.25)
    sensor_cam_eye_pos = [0.316, 0.260, 0.407 + 0.01]
    sensor_cam_target_pos = [0.316, 0.260, 0.01]
    human_cam_eye_pos = [-0.3, 0.6, 0.6]
    human_cam_target_pos = [0.3, 0.3, 0.0]
    max_goal_height = 0.10
    lock_z = True
    target_region_half_size = (0.083, 0.082)
    red_cube_target_region_center = (0.479, 0.25)
    green_cube_target_region_center = (0.121, 0.25)
    robot1_position = [0.481, 0.003, 0]
    robot2_position = [0.121, 0.003, 0]

    def __init__(self, *args, robot_uids=("so101", "so101"), robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos
        )
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos, target=self.human_cam_target_pos
        )
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(
            options, 
            [sapien.Pose(p=self.robot1_position), sapien.Pose(p=self.robot2_position)]
        )

    def _load_scene(self, options: dict):
        self.table_scene = MyTableBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        self.red_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[1, 0, 0, 1],
            name="red_cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        self.green_cube = actors.build_cube(
            self.scene,
            half_size=self.cube_half_size,
            color=[0, 1, 0, 1],
            name="green_cube",
            initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        )
        
    def initialize_agent(self, env_idx: torch.Tensor):
        b = len(env_idx)
        qpos = np.array([-np.pi / 2, 0, 0, np.pi / 2, 0, 0])
            
        qpos1 = (
            self._episode_rng.normal(
                0, self.robot_init_qpos_noise, (b, len(qpos))
            )
            + qpos
        )
        qpos2 = (
            self._episode_rng.normal(
                0, self.robot_init_qpos_noise, (b, len(qpos))
            )
            + qpos
        )
        
        # For MultiAgent, reset each agent separately
        self.agent.agents[0].reset(qpos1)
        self.agent.agents[0].robot.set_pose(sapien.Pose(self.robot1_position))
        
        self.agent.agents[1].reset(qpos2)
        self.agent.agents[1].robot.set_pose(sapien.Pose(self.robot2_position))

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.initialize_agent(env_idx)  
            xyz_1 = torch.zeros((b, 3))
            xyz_2 = torch.zeros((b, 3))
            half_xy = torch.tensor(self.cube_spawn_half_size, device = self.device)
            xyz_1[:, :2] = (
                torch.rand((b, 2)) * half_xy * 2
                - half_xy
            )
            xyz_1[:, 0] += self.cube_spawn_center[0]
            xyz_1[:, 1] += self.cube_spawn_center[1]
            xyz_1[:, 2] = self.cube_half_size

            xyz_2[:, :2] = (
                torch.rand((b, 2)) * half_xy * 2
                - half_xy
            )
            xyz_2[:, 0] += self.cube_spawn_center[0]
            xyz_2[:, 1] += self.cube_spawn_center[1]
            xyz_2[:, 2] = self.cube_half_size
            qs_1 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=self.lock_z)
            qs_2 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=self.lock_z)
            self.red_cube.set_pose(Pose.create_from_pq(xyz_1, qs_1))
            self.green_cube.set_pose(Pose.create_from_pq(xyz_2, qs_2))
            

    def _get_obs_extra(self, info: Dict):
        # Target positions for each cube
        red_target = torch.tensor(
            [[self.red_cube_target_region_center[0], self.red_cube_target_region_center[1], self.cube_half_size]], 
            device=self.device
        ).repeat(self.num_envs, 1)
        green_target = torch.tensor(
            [[self.green_cube_target_region_center[0], self.green_cube_target_region_center[1], self.cube_half_size]], 
            device=self.device
        ).repeat(self.num_envs, 1)
        
        obs = dict(
            is_red_grasped=info["is_red_grasped"],
            is_green_grasped=info["is_green_grasped"],
            robot1_tcp_pose=self.agent.agents[0].tcp_pose.raw_pose,
            robot2_tcp_pose=self.agent.agents[1].tcp_pose.raw_pose,
            red_target_pos=red_target,
            green_target_pos=green_target,
        )
        if "state" in self.obs_mode:
            obs.update(
                red_cube_pose=self.red_cube.pose.raw_pose,
                green_cube_pose=self.green_cube.pose.raw_pose,
            )
        return obs

    def evaluate(self):
        # Check if red cube is in red target region
        red_target_region_x = [self.red_cube_target_region_center[0] - self.target_region_half_size[0], self.red_cube_target_region_center[0] + self.target_region_half_size[0]]
        red_target_region_y = [self.red_cube_target_region_center[1] - self.target_region_half_size[1], self.red_cube_target_region_center[1] + self.target_region_half_size[1]]
        is_red_in_target_region = (self.red_cube.pose.p[:, 0] > red_target_region_x[0]) & (self.red_cube.pose.p[:, 0] < red_target_region_x[1]) & (self.red_cube.pose.p[:, 1] > red_target_region_y[0]) & (self.red_cube.pose.p[:, 1] < red_target_region_y[1])
        is_red_sorted = is_red_in_target_region
        
        # Check if green cube is in green target region
        green_target_region_x = [self.green_cube_target_region_center[0] - self.target_region_half_size[0], self.green_cube_target_region_center[0] + self.target_region_half_size[0]]
        green_target_region_y = [self.green_cube_target_region_center[1] - self.target_region_half_size[1], self.green_cube_target_region_center[1] + self.target_region_half_size[1]]
        is_green_in_target_region = (self.green_cube.pose.p[:, 0] > green_target_region_x[0]) & (self.green_cube.pose.p[:, 0] < green_target_region_x[1]) & (self.green_cube.pose.p[:, 1] > green_target_region_y[0]) & (self.green_cube.pose.p[:, 1] < green_target_region_y[1])
        is_green_sorted = is_green_in_target_region
        
        # Check if both robots are static
        is_robot1_static = self.agent.agents[0].is_static(0.2)
        is_robot2_static = self.agent.agents[1].is_static(0.2)
        is_robots_static = is_robot1_static & is_robot2_static
        
        return {
            "success": is_red_sorted & is_green_sorted & is_robots_static,
            "is_red_sorted": is_red_sorted,
            "is_green_sorted": is_green_sorted,
            "is_robots_static": is_robots_static,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Dense reward for dual-arm cube sorting task.
        
        Design principle: 
        - Robot1 (right side) is responsible for moving RED cube to red target region
        - Robot2 (left side) is responsible for moving GREEN cube to green target region
        - Robots can either grasp or push the cubes (no grasping required)
        
        This assignment is based on workspace reachability of each arm.
        """
        reward = torch.zeros(self.num_envs, device=self.device)
        
        target_green_region_x = self.green_cube_target_region_center[0] + self.target_region_half_size[0]
        target_red_region_x = self.red_cube_target_region_center[0] - self.target_region_half_size[0]
        
        # Get TCP positions for both robots
        robot1_tcp_pos = self.agent.agents[0].tcp_pose.p  # Robot1 handles red cube
        robot2_tcp_pos = self.agent.agents[1].tcp_pose.p  # Robot2 handles green cube
        
        # Get cube positions
        red_cube_pos = self.red_cube.pose.p
        green_cube_pos = self.green_cube.pose.p
        
        # ============ Robot1 - Red Cube Rewards ============
        # Stage 1: Reaching reward - Robot1 approaches red cube (to push or grasp)
        robot1_to_red_dist = torch.linalg.norm(red_cube_pos - robot1_tcp_pos, axis=1)
        reaching_reward_red = 1 - torch.tanh(5 * robot1_to_red_dist)
        
        # Stage 2: Progress reward - Red cube moves towards red target
        red_to_target_dist = torch.max(red_cube_pos[:, 0] - target_red_region_x, 0.0)
        progress_reward_red = 1 - torch.tanh(5 * red_to_target_dist)
        
        # Combine: encourage approaching cube first, then moving cube to target
        # When cube is far from target, prioritize reaching; when close, prioritize progress
        reward += reaching_reward_red * 1.0 + progress_reward_red * 3.0
        
        # Stage 3: Placement bonus - Red cube at target
        is_red_at_target = (red_cube_pos[:, 0] > target_red_region_x)
        reward += is_red_at_target * 3.0
        
        # ============ Robot2 - Green Cube Rewards ============
        # Stage 1: Reaching reward - Robot2 approaches green cube (to push or grasp)
        robot2_to_green_dist = torch.linalg.norm(green_cube_pos - robot2_tcp_pos, axis=1)
        reaching_reward_green = 1 - torch.tanh(5 * robot2_to_green_dist)
        
        # Stage 2: Progress reward - Green cube moves towards green target
        green_to_target_dist = torch.max(green_cube_pos[:, 0] - target_green_region_x, 0.0)
        progress_reward_green = 1 - torch.tanh(5 * green_to_target_dist)
        
        # Combine: encourage approaching cube first, then moving cube to target
        reward += reaching_reward_green * 1.0 + progress_reward_green * 3.0
        
        # Stage 3: Placement bonus - Green cube at target
        is_green_at_target = (green_cube_pos[:, 0] < target_green_region_x).float()
        reward += is_green_at_target * 3.0
        
        # ============ Penalty for wrong robot-cube assignment ============
        # Penalize Robot1 for approaching green cube (should focus on red cube)
        # Use smaller coefficient to ensure correct approach still yields positive net reward
        # Max penalty = 0.1 * 1.0 = 0.1, while correct reaching reward can be up to 1.0
        robot1_to_green_dist = torch.linalg.norm(green_cube_pos - robot1_tcp_pos, axis=1)
        wrong_reach_penalty_1 = torch.clamp(0.1 - robot1_to_green_dist, min=0) * 3.0
        reward -= wrong_reach_penalty_1
        
        # Penalize Robot2 for approaching red cube (should focus on green cube)
        robot2_to_red_dist = torch.linalg.norm(red_cube_pos - robot2_tcp_pos, axis=1)
        wrong_reach_penalty_2 = torch.clamp(0.1 - robot2_to_red_dist, min=0) * 3.0
        reward -= wrong_reach_penalty_2
        
        # ============ Static reward when both cubes are sorted ============
        both_sorted = info["is_red_sorted"] & info["is_green_sorted"]
        qvel1 = self.agent.agents[0].robot.get_qvel()[..., :-1]
        qvel2 = self.agent.agents[1].robot.get_qvel()[..., :-1]
        static_reward_1 = 1 - torch.tanh(5 * torch.linalg.norm(qvel1, axis=1))
        static_reward_2 = 1 - torch.tanh(5 * torch.linalg.norm(qvel2, axis=1))
        reward += (static_reward_1 + static_reward_2) * both_sorted.float()
        
        # ============ Success bonus ============
        reward[info["success"]] = 20.0
        
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        # Max reward is approximately 20 (success bonus)
        # Normalize to [0, 1] range
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 20.0