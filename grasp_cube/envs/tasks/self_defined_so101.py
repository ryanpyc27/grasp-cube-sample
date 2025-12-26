from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

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


@register_env("SelfDefinedSO101-v1", max_episode_steps=200)
class SelfDefinedSO101Env(BaseEnv):
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
    cube_spawn_half_size = (0.083, 0.082)
    cube_spawn_center = (0.121, 0.25)
    sensor_cam_eye_pos = [0.316, 0.260, 0.407 + 0.01]
    sensor_cam_target_pos = [0.316, 0.260, 0.01]
    human_cam_eye_pos = [-0.3, -0.6, 0.6]
    human_cam_target_pos = [0.3, 0.3, 0.0]
    max_goal_height = 0.10
    lock_z = True
    target_region_half_size = (0.083, 0.082)
    red_cube_target_region_center = (0.121, 0.25)
    robot1_position = [0.300, 0.040, 0.01]
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
        # Load cabinet from PartNet-Mobility dataset
        urdf_path = "/homes/yichengp/grasp-cube-sample/partnet-mobility-dataset/45290/mobility.urdf"
        loader = self.scene.create_urdf_loader()
        loader.fix_root_link = True  # Fix the cabinet base
        loader.scale = 0.3  # Scale down the cabinet to 20% of original size
        self.cabinet = loader.load(urdf_path)
        
        cabinet_rotation = euler2quat(0, 0, np.pi / 2.0)
        # Set a temporary pose to compute bounding box
        self.cabinet.set_pose(sapien.Pose(p=[0.3, 0.3, 0.5], q=cabinet_rotation))
        
        # Check if cabinet has movable joints (drawers)
        print(f"\n=== Cabinet Joints Info ===")
        all_joints = self.cabinet.get_active_joints()
        print(f"Number of joints: {len(all_joints)}")
        for i, joint in enumerate(all_joints):
            joint_type = joint.type if isinstance(joint.type, str) else joint.type[0] if len(joint.type) > 0 else 'unknown'
            limits = joint.get_limits()
            print(f"Joint {i}: {joint.name}, type: {joint_type}, limits: {limits}")
        
        # Store drawer joints for later use (prismatic joints are sliding drawers)
        self.drawer_joints = []
        for joint in all_joints:
            joint_type = joint.type if isinstance(joint.type, str) else joint.type[0] if len(joint.type) > 0 else None
            if joint_type in ['prismatic', 'revolute']:
                self.drawer_joints.append(joint)
        print(f"Found {len(self.drawer_joints)} movable drawer joints")
        
        # Remove all resistance from drawer joints to make them freely movable
        for i, joint in enumerate(self.drawer_joints):
            # Set drive properties - use stiffness to hold position once moved
            joint.set_drive_properties(stiffness=1000, damping=50)  # Stiffness to hold position
            # Set friction to 0 to eliminate resistance when actively moving
            joint.set_friction(0.0)
            # Set drive target to current position (no force trying to return to 0)
            joint.set_drive_target(0)
            joint.set_drive_velocity_target(0)
            print(f"Set drawer joint {i} properties: stiffness=1000, damping=50, friction=0")
        
        # Place cabinet on table
        # The cabinet's URDF origin is not at its bottom, so we need to add an offset
        # For this cabinet (45290) at scale 0.2, the origin is roughly at the center
        # After testing, the appropriate z-offset to place it on the table is approximately 0.06m
        # This accounts for the cabinet height and origin position
        
        table_height = 0.01  # Table surface height
        cabinet_z_offset = 0.2  # Empirical offset for this cabinet model (scale 0.2)
        
        desired_cabinet_x = 0.3
        desired_cabinet_y = 0.46
        desired_cabinet_z = table_height + cabinet_z_offset
        
        print(f"\n=== Cabinet Placement ===")
        print(f"Position: x={desired_cabinet_x:.3f}, y={desired_cabinet_y:.3f}, z={desired_cabinet_z:.3f}")
        print(f"Rotation: 90° clockwise around z-axis")
        print(f"Scale: {loader.scale}")
        
        # Store the properly adjusted pose for later use
        self.cabinet_base_pose = sapien.Pose(
            p=[desired_cabinet_x, desired_cabinet_y, desired_cabinet_z], 
            q=cabinet_rotation
        )
        self.cabinet.set_pose(self.cabinet_base_pose)
        
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
    
    def set_drawer_position(self, drawer_idx: int, position: float):
        """
        Set the position of a drawer.
        
        Args:
            drawer_idx: Index of the drawer (0, 1, or 2)
            position: Position in meters (0.0 = closed, max = fully open)
                     Each drawer can open up to 0.14m based on joint limits
        """
        if drawer_idx < len(self.drawer_joints):
            joint = self.drawer_joints[drawer_idx]
            limits = joint.get_limits()
            # Clamp position within limits
            position = np.clip(position, limits[0, 0].item(), limits[0, 1].item())
            # Set joint position
            qpos = self.cabinet.get_qpos()
            joint_idx = self.cabinet.get_active_joints().index(joint)
            qpos[joint_idx] = position
            self.cabinet.set_qpos(qpos)
        else:
            print(f"Warning: drawer_idx {drawer_idx} out of range. Available drawers: {len(self.drawer_joints)}")
    
    def open_drawer(self, drawer_idx: int, open_amount: float = 0.14):
        """
        Open a drawer by a specified amount.
        
        Args:
            drawer_idx: Index of the drawer (0, 1, or 2)
            open_amount: How much to open in meters (default: 0.14 = fully open)
        """
        self.set_drawer_position(drawer_idx, open_amount)
    
    def close_drawer(self, drawer_idx: int):
        """Close a drawer (set position to 0)."""
        self.set_drawer_position(drawer_idx, 0.0)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)
            self.table_scene.initialize(env_idx)
            self.initialize_agent(env_idx)  
            xyz_1 = torch.zeros((b, 3))
            half_xy = torch.tensor(self.cube_spawn_half_size, device = self.device)
            xyz_1[:, :2] = (
                torch.rand((b, 2)) * half_xy * 2
                - half_xy
            )
            xyz_1[:, 0] += self.cube_spawn_center[0]
            xyz_1[:, 1] += self.cube_spawn_center[1]
            xyz_1[:, 2] = self.cube_half_size + 0.01

            qs_1 = randomization.random_quaternions(b, lock_x=True, lock_y=True, lock_z=self.lock_z)
            self.red_cube.set_pose(Pose.create_from_pq(xyz_1, qs_1))
            # Reset cabinet to its properly calculated pose
            self.cabinet.set_pose(self.cabinet_base_pose)
            
            # Reset drawer joint properties to ensure no spring-back force
            for joint in self.drawer_joints:
                joint.set_drive_properties(stiffness=1000, damping=50)
                joint.set_friction(0.0)
                joint.set_drive_target(0)
                joint.set_drive_velocity_target(0)
            
            # Reset all drawers to closed position
            # In batched environments, qpos shape is (batch_size, num_joints)
            current_qpos = self.cabinet.get_qpos()
            if len(current_qpos.shape) == 2:  # Batched: (batch_size, num_joints)
                # Set all drawers to closed (0.0) for all environments in batch
                current_qpos[:, :] = 0.0
            else:  # Single environment: (num_joints,)
                current_qpos[:] = 0.0
            self.cabinet.set_qpos(current_qpos)


    def _get_obs_extra(self, info: Dict):
        # Target positions for each cube
        red_target = torch.tensor(
            [[self.red_cube_target_region_center[0], self.red_cube_target_region_center[1], self.cube_half_size]], 
            device=self.device
        ).repeat(self.num_envs, 1)
        
        obs = dict(
            robot1_tcp_pose=self.agent.agents[0].tcp_pose.raw_pose,
            robot2_tcp_pose=self.agent.agents[1].tcp_pose.raw_pose,
            red_target_pos=red_target,
        )
        if "state" in self.obs_mode:
            obs.update(
                red_cube_pose=self.red_cube.pose.raw_pose,
            )
        return obs

    def evaluate(self):
        """
        Simplified evaluation using cabinet's bounding box.
        The cube should be inside the cabinet (within its bounding box).
        """
        # Get cabinet's global bounding box
        # The cabinet is an articulation, so we need to compute its overall bounding box
        # considering all links at their current positions
        cabinet_pose = self.cabinet.get_pose()
        cabinet_pos = cabinet_pose.p  # (batch_size, 3)
        
        # Get red cube position
        cube_pos = self.red_cube.pose.p  # (batch_size, 3)
        
        # Define cabinet bounding box based on the model
        # From bounding_box.json: min: [-0.455902, -0.801925, -0.540979], max: [0.444886, 0.67794, 0.579644]
        # After scaling by 0.2 and considering the cabinet pose
        scale = 0.2
        cabinet_bbox_min_local = torch.tensor([-0.455902, -0.801925, -0.540979], device=self.device) * scale
        cabinet_bbox_max_local = torch.tensor([0.444886, 0.67794, 0.579644], device=self.device) * scale
        
        # Transform to world coordinates
        # Since the cabinet is rotated 90° around z-axis, we need to rotate the bounding box
        # After 90° rotation: (x, y, z) -> (-y, x, z)
        # So the bounding box in world frame is:
        # min_x_world = cabinet_x + min_y_local (due to 90° rotation)
        # min_y_world = cabinet_y - max_x_local (due to 90° rotation)
        # max_x_world = cabinet_x + max_y_local
        # max_y_world = cabinet_y - min_x_local
        
        # For simplicity and to handle batched environments, compute bounding box in world frame
        if cabinet_pos.dim() == 1:
            cabinet_pos = cabinet_pos.unsqueeze(0)
        
        # After 90° rotation around z-axis: swap x and y, negate new x
        bbox_min_x = cabinet_pos[:, 0] + cabinet_bbox_min_local[1] * scale  # -y becomes x
        bbox_max_x = cabinet_pos[:, 0] + cabinet_bbox_max_local[1] * scale
        bbox_min_y = cabinet_pos[:, 1] - cabinet_bbox_max_local[0] * scale  # -x becomes y
        bbox_max_y = cabinet_pos[:, 1] - cabinet_bbox_min_local[0] * scale
        bbox_min_z = cabinet_pos[:, 2] + cabinet_bbox_min_local[2] * scale
        bbox_max_z = cabinet_pos[:, 2] + cabinet_bbox_max_local[2] * scale
        
        # Check if cube is inside the bounding box
        is_in_bbox_x = (cube_pos[:, 0] >= bbox_min_x) & (cube_pos[:, 0] <= bbox_max_x)
        is_in_bbox_y = (cube_pos[:, 1] >= bbox_min_y) & (cube_pos[:, 1] <= bbox_max_y)
        is_in_bbox_z = (cube_pos[:, 2] >= bbox_min_z) & (cube_pos[:, 2] <= bbox_max_z)
        
        is_cube_in_cabinet = is_in_bbox_x & is_in_bbox_y & is_in_bbox_z
        
        # For reward shaping: compute a target position (center of the cabinet's lowest drawer region)
        # Using the center of the bottom part of the cabinet
        target_x = (bbox_min_x + bbox_max_x) / 2
        target_y = (bbox_min_y + bbox_max_y) / 2
        target_z = bbox_min_z + (bbox_max_z - bbox_min_z) * 0.2  # Lower 20% of cabinet height
        
        drawer_target_pos = torch.stack([target_x, target_y, target_z], dim=1)
        placed_threshold = 0.15  # Lenient threshold for reward shaping
        red_placed = torch.linalg.norm(cube_pos - drawer_target_pos, axis=1) < placed_threshold
        
        # Check if both robots are static
        is_robot1_static = self.agent.agents[0].is_static(0.2)
        is_robot2_static = self.agent.agents[1].is_static(0.2)
        is_robots_static = is_robot1_static & is_robot2_static
        
        return {
            "success": is_cube_in_cabinet & is_robots_static,
            "is_red_sorted": is_cube_in_cabinet,
            "red_placed": red_placed,
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
        
        # ============ Reaching Rewards ============
        # Compute distances from both TCPs to both cubes
        left_to_red = torch.linalg.norm(red_cube_pos - left_tcp, axis=1)
        right_to_red = torch.linalg.norm(red_cube_pos - right_tcp, axis=1)
        
        # Minimum distance to each cube (from either robot)
        red_dist = torch.minimum(left_to_red, right_to_red)
        
        # Tanh-based reach rewards
        red_reach = 1.0 - torch.tanh(5.0 * red_dist)
        
        # Once a cube is placed (more lenient than sorted), keep its reach term at maximum to avoid reward drops
        # This allows robots to disengage and move away once cube is roughly in place
        red_reach_term = torch.where(
            info["red_placed"].bool(),
            torch.ones_like(red_reach),
            red_reach
        )
        
        # Use a soft "choose-one" reach term to avoid pulling both arms toward the midpoint
        # This encourages focusing on one cube at a time
        reaching_reward = red_reach_term
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
        
        # Only give placement reward if:
        # 1. Cube is not yet sorted
        # 2. At least one robot is close to the cube (actively engaging with it)
        red_is_engaged = (red_dist < close_threshold).float()
        
        red_placement_term = red_placement * (~info["is_red_sorted"].bool()).float() * red_is_engaged
        
        reward += red_placement_term * 2.0
        
        # ============ Static Reward ============
        # Encourage robots to be static when both cubes are sorted
        both_sorted = info["is_red_sorted"]
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