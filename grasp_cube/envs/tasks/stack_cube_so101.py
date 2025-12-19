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


@register_env("StackCubeSO101-v1", max_episode_steps=50)
class StackCubeSO101Env(BaseEnv):
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
        "so101",
    ]
    agent: SO101
    cube_half_size = 0.015
    goal_thresh = 0.015 * 1.25
    cube_spawn_half_size = (0.083, 0.082)
    cube_spawn_center = (0.479, 0.25)
    sensor_cam_eye_pos = [0.316, 0.260, 0.407 + 0.01]
    sensor_cam_target_pos = [0.316, 0.260, 0.01]
    human_cam_eye_pos = [-0.3, 0.6, 0.6]
    human_cam_target_pos = [0.3, 0.3, 0.0]
    max_goal_height = 0.10
    lock_z = True

    def __init__(self, *args, robot_uids="so101", robot_init_qpos_noise=0.02, **kwargs):
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
        super()._load_agent(options, sapien.Pose(p=[0.481, 0.003, 0]))

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
        qpos = (
            self._episode_rng.normal(
                0, self.robot_init_qpos_noise, (b, len(qpos))
            )
            + qpos
        )
        self.agent.reset(qpos)
        self.agent.robot.set_pose(
            sapien.Pose([0.481, 0.003, 0])
        )

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
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        goal_pos = self.green_cube.pose.p.clone()
        goal_pos[:, 2] += 2 * self.cube_half_size
        
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp_pose.raw_pose,
            goal_pos=goal_pos,
        )
        if "state" in self.obs_mode:
            obs.update(
                red_cube_pos=self.red_cube.pose.raw_pose,
                green_cube_pos=self.green_cube.pose.raw_pose,
                tcp_to_obj_pos=self.red_cube.pose.p - self.agent.tcp_pose.p,
                obj_to_goal_pos=goal_pos - self.red_cube.pose.p,
            )
        return obs

    def evaluate(self):
        target_pos = self.green_cube.pose.p.clone()
        target_pos[:, 2] += 2 * self.cube_half_size

        stacking_dist = torch.linalg.norm(self.red_cube.pose.p - target_pos, axis=1)
        is_stacked = stacking_dist < self.goal_thresh


        is_grasped = self.agent.is_grasping(self.red_cube)
        is_robot_static = self.agent.is_static(0.2)
        return {
            "success": is_stacked & is_robot_static,
            "is_stacked": is_stacked,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        is_grasped = info["is_grasped"]
        
        tcp_to_obj_dist = torch.linalg.norm(
            self.red_cube.pose.p - self.agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_obj_dist)
        reward = reaching_reward * (1 - is_grasped * 0.5)
        
        reward += is_grasped
        
        table_top_z = 0.01
        cube_height = self.red_cube.pose.p[:, 2] - table_top_z
        lift_reward = torch.clamp(cube_height / 0.05, 0, 1)
        reward += lift_reward * is_grasped * 2
        
        xy_dist = torch.linalg.norm(
            self.red_cube.pose.p[:, :2] - self.green_cube.pose.p[:, :2], axis=1
        )
        xy_diff_reward = 1 - torch.tanh(10 * xy_dist)
        reward += xy_diff_reward * is_grasped * 2
        
        z_diff_reward = 1 - torch.tanh(10 * torch.abs(self.red_cube.pose.p[:, 2] - self.green_cube.pose.p[:, 2]))
        is_close_enough = xy_dist < 0.02
        is_lifted = self.red_cube.pose.p[:, 2] > self.green_cube.pose.p[:, 2]
        reward += z_diff_reward * is_close_enough * is_lifted * 2
        
        qvel = self.agent.robot.get_qvel()
        qvel = qvel[..., :-1]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_stacked"]
        
        reward[info["success"]] = 10
        
        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 10