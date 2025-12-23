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
    cube_spawn_half_size = (0.063, 0.067)
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
    robot1_position = [0.481, 0.080, 0.01]
    robot2_position = [0.119, 0.080, 0.01]

    def __init__(self, *args, robot_uids=("so101", "so101"), robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(
            eye=self.sensor_cam_eye_pos, target=self.sensor_cam_target_pos, 
            up = [0, 1, 0]
        )
        # Add base camera and wrist cameras for both robots
        return [
            CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100),
            CameraConfig(
                "wrist_camera_1",
                sapien.Pose(),  # The pose is relative to the mount link
                128,
                128,
                np.pi / 2,
                0.01,
                100,
                mount=self.agent.agents[0].robot.links_map["camera_link"],
            ),
            CameraConfig(
                "wrist_camera_2",
                sapien.Pose(),  # The pose is relative to the mount link
                128,
                128,
                np.pi / 2,
                0.01,
                100,
                mount=self.agent.agents[1].robot.links_map["camera_link"],
            ),
        ]

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
        qpos = np.array([0, 0, 0, np.pi / 2, 0, 0])
        quaternion = [0.7071068, 0, 0, 0.7071068]
            
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
        self.agent.agents[0].robot.set_pose(sapien.Pose(self.robot1_position, q=quaternion))
        
        self.agent.agents[1].reset(qpos2)
        self.agent.agents[1].robot.set_pose(sapien.Pose(self.robot2_position, q=quaternion))

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
            xyz_1[:, 2] = self.cube_half_size + 0.01

            xyz_2[:, :2] = (
                torch.rand((b, 2)) * half_xy * 2
                - half_xy
            )
            xyz_2[:, 0] += self.cube_spawn_center[0]
            xyz_2[:, 1] += self.cube_spawn_center[1]
            xyz_2[:, 2] = self.cube_half_size + 0.01
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
        
        # Check if cubes are "placed" (close enough to target, more lenient than sorted)
        # This is used for reward shaping to allow robots to disengage earlier
        red_target_pos = torch.tensor(
            [[self.red_cube_target_region_center[0], self.red_cube_target_region_center[1], self.cube_half_size]], 
            device=self.device
        ).repeat(self.num_envs, 1)
        green_target_pos = torch.tensor(
            [[self.green_cube_target_region_center[0], self.green_cube_target_region_center[1], self.cube_half_size]], 
            device=self.device
        ).repeat(self.num_envs, 1)
        
        placed_threshold = max(self.target_region_half_size) * 0.9  # More lenient than sorted
        red_placed = torch.linalg.norm(self.red_cube.pose.p - red_target_pos, axis=1) < placed_threshold
        green_placed = torch.linalg.norm(self.green_cube.pose.p - green_target_pos, axis=1) < placed_threshold
        
        # Check if both robots are static
        is_robot1_static = self.agent.agents[0].is_static(0.2)
        is_robot2_static = self.agent.agents[1].is_static(0.2)
        is_robots_static = is_robot1_static & is_robot2_static
        
        return {
            "success": is_red_sorted & is_green_sorted & is_robots_static,
            "is_red_sorted": is_red_sorted,
            "is_green_sorted": is_green_sorted,
            "red_placed": red_placed,
            "green_placed": green_placed,
            "is_robots_static": is_robots_static,
        }

    def _compute_reach_reward(self, cube_pos: torch.Tensor, tcp_positions: list) -> torch.Tensor:
        """
        Compute reach reward for a cube using minimum distance from any TCP.
        
        Args:
            cube_pos: Cube position tensor [batch_size, 3]
            tcp_positions: List of TCP position tensors, each [batch_size, 3]
            
        Returns:
            Reach reward tensor [batch_size]
        """
        # Compute distances from all TCPs to the cube
        distances = []
        for tcp_pos in tcp_positions:
            dist = torch.linalg.norm(cube_pos - tcp_pos, axis=1)
            distances.append(dist)
        
        # Use minimum distance (closest TCP to cube)
        min_dist = torch.minimum(distances[0], distances[1])
        
        # Tanh-based reach reward: higher when closer, saturates smoothly
        reach_reward = 1.0 - torch.tanh(5.0 * min_dist)
        return reach_reward
    
    def _compute_placement_reward(
        self, 
        cube_pos: torch.Tensor, 
        target_center: tuple, 
        target_half_size: tuple
    ) -> torch.Tensor:
        """
        Compute placement reward based on how close cube is to target region.
        
        Args:
            cube_pos: Cube position tensor [batch_size, 3]
            target_center: Target region center (x, y)
            target_half_size: Target region half size (x, y)
            
        Returns:
            Placement reward tensor [batch_size]
        """
        # Compute distance to target center in XY plane
        target_xy = torch.tensor(
            [[target_center[0], target_center[1]]], 
            device=self.device
        ).repeat(self.num_envs, 1)
        
        xy_dist = torch.linalg.norm(cube_pos[:, :2] - target_xy, axis=1)
        
        # Normalize by target region size (use max half_size as normalization)
        max_half_size = max(target_half_size)
        normalized_dist = xy_dist / (max_half_size + 1e-6)
        
        # Reward decreases smoothly as distance increases
        placement_reward = 1.0 - torch.tanh(3.0 * normalized_dist)
        return placement_reward
    
    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        """
        Dense reward for dual-arm cube sorting task.
        
        Design principles:
        1. Flexible assignment: Both robots can reach either cube, using minimum distance
        2. Soft choose-one: Use maximum of reach terms to avoid both arms going to midpoint
        3. Preserve progress: Once a cube is sorted, keep its reach term at maximum
        4. Placement guidance: Additional reward for moving cubes toward target regions
        
        This design allows the policy to learn optimal robot-cube assignments naturally.
        """
        reward = torch.zeros(self.num_envs, device=self.device)
        
        # Get TCP positions for both robots
        left_tcp = self.agent.agents[0].tcp_pose.p
        right_tcp = self.agent.agents[1].tcp_pose.p
        tcp_positions = [left_tcp, right_tcp]
        
        # Get cube positions
        red_cube_pos = self.red_cube.pose.p
        green_cube_pos = self.green_cube.pose.p
        
        # ============ Reaching Rewards ============
        # Compute distances from both TCPs to both cubes
        left_to_red = torch.linalg.norm(red_cube_pos - left_tcp, axis=1)
        right_to_red = torch.linalg.norm(red_cube_pos - right_tcp, axis=1)
        left_to_green = torch.linalg.norm(green_cube_pos - left_tcp, axis=1)
        right_to_green = torch.linalg.norm(green_cube_pos - right_tcp, axis=1)
        
        # Minimum distance to each cube (from either robot)
        red_dist = torch.minimum(left_to_red, right_to_red)
        green_dist = torch.minimum(left_to_green, right_to_green)
        
        # Tanh-based reach rewards
        red_reach = 1.0 - torch.tanh(5.0 * red_dist)
        green_reach = 1.0 - torch.tanh(5.0 * green_dist)
        
        # Once a cube is placed (more lenient than sorted), keep its reach term at maximum to avoid reward drops
        # This allows robots to disengage and move away once cube is roughly in place
        red_reach_term = torch.where(
            info["red_placed"].bool(),
            torch.ones_like(red_reach),
            red_reach
        )
        green_reach_term = torch.where(
            info["green_placed"].bool(),
            torch.ones_like(green_reach),
            green_reach
        )
        
        # Use a soft "choose-one" reach term to avoid pulling both arms toward the midpoint
        # This encourages focusing on one cube at a time
        reaching_reward = torch.maximum(red_reach_term, green_reach_term)
        reward += reaching_reward * 2.0  # Scale factor for reach reward
        
        # ============ Placement Rewards ============
        # CRITICAL: Only give placement reward when robot is CLOSE to cube
        # This prevents the policy from learning to "stay away to preserve placement reward"
        
        # Define "close" threshold (robot must be within this distance to get placement reward)
        close_threshold = 0.08  # 8cm
        
        red_placement = self._compute_placement_reward(
            red_cube_pos,
            self.red_cube_target_region_center,
            self.target_region_half_size
        )
        green_placement = self._compute_placement_reward(
            green_cube_pos,
            self.green_cube_target_region_center,
            self.target_region_half_size
        )
        
        # Only give placement reward if:
        # 1. Cube is not yet sorted
        # 2. At least one robot is close to the cube (actively engaging with it)
        red_is_engaged = (red_dist < close_threshold).float()
        green_is_engaged = (green_dist < close_threshold).float()
        
        red_placement_term = red_placement * (~info["is_red_sorted"].bool()).float() * red_is_engaged
        green_placement_term = green_placement * (~info["is_green_sorted"].bool()).float() * green_is_engaged
        
        reward += (red_placement_term + green_placement_term) * 2.0
        
        # ============ Static Reward ============
        # Encourage robots to be static when both cubes are sorted
        both_sorted = info["is_red_sorted"] & info["is_green_sorted"]
        if both_sorted.any():
            qvel1 = self.agent.agents[0].robot.get_qvel()[..., :-1]
            qvel2 = self.agent.agents[1].robot.get_qvel()[..., :-1]
            static_reward_1 = 1.0 - torch.tanh(5.0 * torch.linalg.norm(qvel1, axis=1))
            static_reward_2 = 1.0 - torch.tanh(5.0 * torch.linalg.norm(qvel2, axis=1))
            static_reward = (static_reward_1 + static_reward_2) / 2.0
            reward += static_reward * both_sorted.float() * 1.0
        
        # ============ Success Bonus ============
        # Large bonus for complete success
        reward[info["success"]] = 10.0
        
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        """
        Normalized dense reward in [0, 1] range.
        
        Max reward breakdown:
        - Reaching reward: 1.0 * 2 = 2.0 (sum of both reach terms)
        - Placement reward: 2.0 * 2 = 4.0 (max for both cubes when engaged)
        - Static reward: 1.0 (when both sorted)
        - Success bonus: 10.0
        Total max: ~17.0
        """
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 17.0