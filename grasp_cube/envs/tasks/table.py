import os.path as osp
from pathlib import Path

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.robots.fetch import FETCH_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder


class MyTableBuilder(SceneBuilder):
    """
    A simple scene builder that adds a table to the scene such that the height of the table is at 0.1, and
    gives reasonable initial poses for robots.
    We add bands on the table to fit the real robot setting. 
    """

    def build(self):
        builder = self.scene.create_actor_builder()
        model_dir = Path(osp.dirname(__file__)) / "assets"

        # add table
        self.table_pose = sapien.Pose(p = [0.3, 0.3, 0.01 / 2], q=euler2quat(0, 0, 0))
        self.table_q = euler2quat(0, 0, 0)
        table_half_size = (0.6 / 2, 0.6 / 2, 0.01 / 2)
        builder.add_box_visual(
            pose=self.table_pose,
            half_size = table_half_size, 
            material=sapien.render.RenderMaterial(base_color=[0.9, 0.9, 0.9, 1.0]), 
        )
        builder.add_box_collision(
            pose=self.table_pose,
            half_size=table_half_size,
        )
        builder.initial_pose = sapien.Pose(
            p=[0, 0, 0], q=self.table_q
        )
        table = builder.build_kinematic(name="table-workspace")

        # add bands
        table_height = self.table_pose.get_p()[2]
        longband_builder = self.scene.create_actor_builder()
        longband_builder.add_box_visual(half_size=[0.28, 0.009, 0.01], material=[0, 0, 0])
        longband_builder.add_box_collision(half_size=[0.28, 0.009, 0.01])
        
        longband_up = longband_builder.build_static(name="longband_up")
        longband_up.set_pose(sapien.Pose([0.30, 0.341, table_height + 0.005]))
        longband_down = longband_builder.build_static(name="longband_down")
        longband_down.set_pose(sapien.Pose([0.30, 0.159, table_height + 0.005]))

        shortband_builder_up = self.scene.create_actor_builder()
        shortband_builder_up.add_box_visual(half_size=[0.009, 0.082, 0.01], material=[0, 0, 0])
        shortband_builder_up.add_box_collision(half_size=[0.009, 0.082, 0.01])

        shortband_up1 = shortband_builder_up.build_static(name="shortband_up1")
        shortband_up1.set_pose(sapien.Pose([0.029, 0.25, table_height + 0.005]))
        shortband_up2 = shortband_builder_up.build_static(name="shortband_up2")
        shortband_up2.set_pose(sapien.Pose([0.213, 0.25, table_height + 0.005]))
        shortband_up3 = shortband_builder_up.build_static(name="shortband_up3")
        shortband_up3.set_pose(sapien.Pose([0.387, 0.25, table_height + 0.005]))
        shortband_up4 = shortband_builder_up.build_static(name="shortband_up4")
        shortband_up4.set_pose(sapien.Pose([0.571, 0.25, table_height + 0.005]))

        shortband_builder_down = self.scene.create_actor_builder()
        shortband_builder_down.add_box_visual(half_size=[0.009, 0.075, 0.01], material=[0, 0, 0])
        shortband_builder_down.add_box_collision(half_size=[0.009, 0.075, 0.01])
        
        shortband_down1 = shortband_builder_down.build_static(name="shortband_down1")
        shortband_down1.set_pose(sapien.Pose([0.213, 0.075, table_height + 0.005]))
        shortband_down2 = shortband_builder_down.build_static(name="shortband_down2")
        shortband_down2.set_pose(sapien.Pose([0.387, 0.075, table_height + 0.005]))


        floor_width = 100
        if self.scene.parallel_in_single_scene:
            floor_width = 500
        self.ground = build_ground(
            self.scene, floor_width=floor_width, altitude=0
        )
        self.table = table
        self.scene_objects: list[sapien.Entity] = [self.table, self.ground]

    def initialize(self, env_idx: torch.Tensor):
        # table_height = 0.9196429
        b = len(env_idx)
        self.table.set_pose(
            sapien.Pose(p=(0, 0, 0), q=euler2quat(0, 0, 0))
        )
        if self.env.robot_uids == "panda":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids == "panda_wristcam":
            # fmt: off
            qpos = np.array(
                [0.0, np.pi / 8, 0, -np.pi * 5 / 8, 0, np.pi * 3 / 4, -np.pi / 4, 0.04, 0.04]
            )
            # fmt: on
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in [
            "xarm6_allegro_left",
            "xarm6_allegro_right",
            "xarm6_robotiq",
            "xarm6_nogripper",
        ]:
            qpos = self.env.agent.keyframes["rest"].qpos
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.522, 0, 0]))
        elif self.env.robot_uids == "fetch":
            qpos = np.array(
                [
                    0,
                    0,
                    0,
                    0.386,
                    0,
                    0,
                    0,
                    -np.pi / 4,
                    0,
                    np.pi / 4,
                    0,
                    np.pi / 3,
                    0,
                    0.015,
                    0.015,
                ]
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-1.05, 0, -self.table_height]))

            self.ground.set_collision_group_bit(
                group=2, bit_idx=FETCH_WHEELS_COLLISION_BIT, bit=1
            )
        elif self.env.robot_uids == ("panda", "panda"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif self.env.robot_uids == ("panda_wristcam", "panda_wristcam"):
            agent: MultiAgent = self.env.agent
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                    0.04,
                    0.04,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            qpos[:, -2:] = 0.04
            agent.agents[1].reset(qpos)
            agent.agents[1].robot.set_pose(
                sapien.Pose([0, 0.75, 0], q=euler2quat(0, 0, -np.pi / 2))
            )
            agent.agents[0].reset(qpos)
            agent.agents[0].robot.set_pose(
                sapien.Pose([0, -0.75, 0], q=euler2quat(0, 0, np.pi / 2))
            )
        elif (
            "dclaw" in self.env.robot_uids
            or "allegro" in self.env.robot_uids
            or "trifinger" in self.env.robot_uids
        ):
            # Need to specify the robot qpos for each sub-scenes using tensor api
            pass
        elif self.env.robot_uids == "panda_stick":
            qpos = np.array(
                [
                    0.0,
                    np.pi / 8,
                    0,
                    -np.pi * 5 / 8,
                    0,
                    np.pi * 3 / 4,
                    np.pi / 4,
                ]
            )
            if self.env._enhanced_determinism:
                qpos = (
                    self.env._batched_episode_rng[env_idx].normal(
                        0, self.robot_init_qpos_noise, len(qpos)
                    )
                    + qpos
                )
            else:
                qpos = (
                    self.env._episode_rng.normal(
                        0, self.robot_init_qpos_noise, (b, len(qpos))
                    )
                    + qpos
                )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(sapien.Pose([-0.615, 0, 0]))
        elif self.env.robot_uids in ["widowxai", "widowxai_wristcam"]:
            qpos = self.env.agent.keyframes["ready_to_grasp"].qpos
            self.env.agent.reset(qpos)
        elif self.env.robot_uids == "so100":
            qpos = np.array([0, 0, 0, np.pi / 2, np.pi / 2, 0])
            qpos = (
                self.env._episode_rng.normal(
                    0, self.robot_init_qpos_noise, (b, len(qpos))
                )
                + qpos
            )
            self.env.agent.reset(qpos)
            self.env.agent.robot.set_pose(
                sapien.Pose([-0.725, 0, 0], q=euler2quat(0, 0, np.pi / 2))
            )
