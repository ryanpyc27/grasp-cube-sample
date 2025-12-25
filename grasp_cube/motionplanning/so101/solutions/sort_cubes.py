import numpy as np
import sapien
import mplib
from transforms3d.euler import euler2quat
from transforms3d import euler

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.utils.structs.pose import to_sapien_pose
from grasp_cube.envs.tasks.sort_cube_so101 import SortCubeSO101Env
from grasp_cube.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


class DualArmSO101MotionPlanner:
    """
    Dual-arm motion planner for SO101 robots.
    Handles two robots independently while stepping the environment with combined actions.
    """
    OPEN = 0.6
    CLOSED = 0
    MOVE_GROUP = "gripper_link_tip"
    
    def __init__(
        self,
        env: SortCubeSO101Env,
        debug: bool = False,
        vis: bool = True,
        print_env_info: bool = False,
        joint_vel_limits: float = 0.6,  # 降低速度限制，提高稳定性
        joint_acc_limits: float = 0.6,  # 降低加速度限制
    ):
        self.env = env
        self.base_env: SortCubeSO101Env = env.unwrapped
        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.elapsed_steps = 0
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        
        # Get agents (robots)
        self.agent1: BaseAgent = self.base_env.agent.agents[0]  # Robot 1 - handles red cube
        self.agent2: BaseAgent = self.base_env.agent.agents[1]  # Robot 2 - handles green cube
        
        self.robot1 = self.agent1.robot
        self.robot2 = self.agent2.robot
        
        # Base poses for each robot
        self.base_pose1 = to_sapien_pose(self.robot1.pose)
        self.base_pose2 = to_sapien_pose(self.robot2.pose)
        
        # Setup planners for both robots
        self.planner1 = self._setup_planner(self.agent1, self.base_pose1)
        self.planner2 = self._setup_planner(self.agent2, self.base_pose2)
        
        self.control_mode = self.base_env.control_mode
        
        # Gripper states for both robots
        self.gripper_state1 = self.OPEN
        self.gripper_state2 = self.OPEN
        
        # Visual grasp pose transform for SO101
        self._so_101_visual_grasp_pose_transform = sapien.Pose(q=euler.euler2quat(0, 0, np.pi / 2))
    
    def _setup_planner(self, agent: BaseAgent, base_pose: sapien.Pose) -> mplib.Planner:
        """Setup motion planner for a single robot."""
        link_names = [link.get_name() for link in agent.robot.get_links()]
        joint_names = [joint.get_name() for joint in agent.robot.get_active_joints()]
        
        planner = mplib.Planner(
            urdf=agent.urdf_path,
            srdf=agent.urdf_path.replace(".urdf", ".srdf"),
            user_link_names=link_names,
            user_joint_names=joint_names,
            move_group=self.MOVE_GROUP,
        )
        planner.set_base_pose(np.hstack([base_pose.p, base_pose.q]))
        planner.joint_vel_limits = np.asarray(planner.joint_vel_limits) * self.joint_vel_limits
        planner.joint_acc_limits = np.asarray(planner.joint_acc_limits) * self.joint_acc_limits
        return planner
    
    def _get_tcp_transform(self, agent: BaseAgent) -> sapien.Pose:
        """Get the TCP transform for SO101 grasp pose planning."""
        return agent.robot.links_map["gripper_link_tip"].pose.sp * agent.tcp_pose.sp.inv()
    
    def _transform_pose_for_planning(self, target: sapien.Pose, agent: BaseAgent) -> sapien.Pose:
        """Transform pose for SO101 planning."""
        tcp_transform = self._get_tcp_transform(agent)
        return sapien.Pose(p=target.p + tcp_transform.p, q=target.q)
    
    def _get_current_qpos(self, robot_idx: int) -> np.ndarray:
        """Get current joint positions for specified robot."""
        if robot_idx == 0:
            return self.robot1.get_qpos().cpu().numpy()[0]
        else:
            return self.robot2.get_qpos().cpu().numpy()[0]
    
    def _step_env(self, action1: np.ndarray, action2: np.ndarray):
        """Step environment with combined actions from both robots."""
        # Combine actions: [robot1_action, robot2_action]
        combined_action = np.concatenate([action1, action2])
        obs, reward, terminated, truncated, info = self.env.step(combined_action)
        self.elapsed_steps += 1
        
        if self.print_env_info:
            print(f"[{self.elapsed_steps:3}] Env Output: reward={reward} info={info}")
        if self.vis:
            self.base_env.render_human()
        
        return obs, reward, terminated, truncated, info
    
    def _make_action(self, qpos: np.ndarray, gripper_state: float) -> np.ndarray:
        """Create action array for a single robot."""
        if self.control_mode == "pd_joint_pos_vel":
            qvel = np.zeros_like(qpos)
            return np.hstack([qpos, qvel, gripper_state])
        else:
            return np.hstack([qpos, gripper_state])
    
    def open_gripper(self, robot_idx: int, t: int = 6):
        """Open gripper for specified robot (0 or 1)."""
        if robot_idx == 0:
            self.gripper_state1 = self.OPEN
        else:
            self.gripper_state2 = self.OPEN
        
        qpos1 = self._get_current_qpos(0)[:len(self.planner1.joint_vel_limits)]
        qpos2 = self._get_current_qpos(1)[:len(self.planner2.joint_vel_limits)]
        
        for _ in range(t):
            action1 = self._make_action(qpos1, self.gripper_state1)
            action2 = self._make_action(qpos2, self.gripper_state2)
            obs, reward, terminated, truncated, info = self._step_env(action1, action2)
        
        return obs, reward, terminated, truncated, info
    
    def close_gripper(self, robot_idx: int, t: int = 12):
        """Close gripper for specified robot (0 or 1)."""
        if robot_idx == 0:
            self.gripper_state1 = self.CLOSED
        else:
            self.gripper_state2 = self.CLOSED
        
        qpos1 = self._get_current_qpos(0)[:len(self.planner1.joint_vel_limits)]
        qpos2 = self._get_current_qpos(1)[:len(self.planner2.joint_vel_limits)]
        
        for _ in range(t):
            action1 = self._make_action(qpos1, self.gripper_state1)
            action2 = self._make_action(qpos2, self.gripper_state2)
            obs, reward, terminated, truncated, info = self._step_env(action1, action2)
        
        return obs, reward, terminated, truncated, info
    
    def move_robot_to_pose(self, robot_idx: int, pose: sapien.Pose, refine_steps: int = 0):
        """
        Move specified robot (0 or 1) to target pose while keeping the other robot still.
        """
        pose = to_sapien_pose(pose)
        
        # Select planner and agent based on robot index
        if robot_idx == 0:
            planner = self.planner1
            agent = self.agent1
        else:
            planner = self.planner2
            agent = self.agent2
        
        # Transform pose for planning
        transformed_pose = self._transform_pose_for_planning(pose, agent)
        
        # Plan path with increased planning time and relaxed constraints
        current_qpos = self._get_current_qpos(robot_idx)
        result = planner.plan_qpos_to_pose(
            np.concatenate([transformed_pose.p, transformed_pose.q]),
            current_qpos,
            time_step=self.base_env.control_timestep,
            use_point_cloud=False,
            wrt_world=True,
            planning_time=10.0,  # 增加planning time，提高IK成功率
        )
        
        if result["status"] != "Success":
            print(f"Robot {robot_idx + 1} planning failed: {result['status']}")
            return -1
        
        # Follow path
        return self._follow_path(robot_idx, result, refine_steps)
    
    def _follow_path(self, robot_idx: int, result: dict, refine_steps: int = 0):
        """
        Follow planned path for specified robot while keeping the other robot still.
        """
        n_step = result["position"].shape[0]
        
        for i in range(n_step + refine_steps):
            target_qpos = result["position"][min(i, n_step - 1)]
            
            # Get current qpos for both robots
            qpos1 = self._get_current_qpos(0)[:len(self.planner1.joint_vel_limits)]
            qpos2 = self._get_current_qpos(1)[:len(self.planner2.joint_vel_limits)]
            
            # Update qpos for the moving robot
            if robot_idx == 0:
                qpos1 = target_qpos
            else:
                qpos2 = target_qpos
            
            action1 = self._make_action(qpos1, self.gripper_state1)
            action2 = self._make_action(qpos2, self.gripper_state2)
            obs, reward, terminated, truncated, info = self._step_env(action1, action2)
        
        return obs, reward, terminated, truncated, info
    
    def close(self):
        """Clean up resources."""
        pass


def compute_grasp_pose(env: SortCubeSO101Env, cube, agent: BaseAgent) -> sapien.Pose:
    """Compute grasp pose for a cube using specified agent."""
    FINGER_LENGTH = 0.025  # Deeper grasp for better grip
    obb = get_actor_obb(cube)
    approaching = np.array([0, 0, -1])
    
    # Rotate around x-axis to align with the expected frame
    tcp_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * agent.tcp_pose.sp
    target_closing = tcp_pose.to_transformation_matrix()[:3, 1]
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    
    # Build grasp pose at cube position
    cube_pos = cube.pose.sp.p
    # Grasp at the center of the cube, not above it
    # This provides better stability
    adjusted_pos = cube_pos.copy()
    # Keep the z position at cube center for better grip
    adjusted_pos[2] = env.unwrapped.cube_half_size + 0.003  # Slightly above cube bottom
    
    grasp_pose = agent.build_grasp_pose(approaching, grasp_info["closing"], adjusted_pos)
    
    # Transform for SO101
    grasp_pose = grasp_pose * sapien.Pose(q=np.array([-1, 0, 0, 1]) / np.sqrt(2))
    
    return grasp_pose


def pick_and_place(planner: DualArmSO101MotionPlanner, robot_idx: int, 
                   cube, target_pos: np.ndarray, env: SortCubeSO101Env):
    """
    Execute pick and place sequence for specified robot and cube.
    
    Args:
        planner: Dual arm motion planner
        robot_idx: 0 for robot1, 1 for robot2
        cube: The cube actor to pick up
        target_pos: Target position [x, y, z] to place the cube
        env: The environment
    """
    agent = planner.agent1 if robot_idx == 0 else planner.agent2
    robot = agent.robot
    
    # Debug: Print positions (convert tensors to numpy for formatting)
    cube_pos = cube.pose.sp.p
    robot_base_pos = robot.pose.p
    tcp_pos = agent.tcp_pose.sp.p
    
    # Convert to numpy/list for printing
    if hasattr(robot_base_pos, 'cpu'):
        robot_base_pos = robot_base_pos.cpu().numpy()[0]
    if hasattr(tcp_pos, 'cpu'):
        tcp_pos = tcp_pos.cpu().numpy()[0] if len(tcp_pos.shape) > 1 else tcp_pos.cpu().numpy()
    if hasattr(cube_pos, 'cpu'):
        cube_pos = cube_pos.cpu().numpy()[0] if len(cube_pos.shape) > 1 else cube_pos.cpu().numpy()
        
    # print(f"\nRobot {robot_idx + 1} Debug Info:")
    # print(f"  - Robot base: [{robot_base_pos[0]:.3f}, {robot_base_pos[1]:.3f}, {robot_base_pos[2]:.3f}]")
    # print(f"  - Current TCP: [{tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f}]")
    # print(f"  - Cube position: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
    # print(f"  - Target position: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    
    # Calculate distances
    distance_to_cube = np.sqrt((cube_pos[0] - robot_base_pos[0])**2 + 
                               (cube_pos[1] - robot_base_pos[1])**2 + 
                               (cube_pos[2] - robot_base_pos[2])**2)
    distance_tcp_to_cube = np.sqrt((cube_pos[0] - tcp_pos[0])**2 + 
                                   (cube_pos[1] - tcp_pos[1])**2 + 
                                   (cube_pos[2] - tcp_pos[2])**2)
    # print(f"  - Distance (base to cube): {distance_to_cube:.3f}m")
    # print(f"  - Distance (TCP to cube): {distance_tcp_to_cube:.3f}m")
    
    # Compute grasp pose for the cube
    grasp_pose = compute_grasp_pose(env, cube, agent)
    # print(f"  - Grasp pose: [{grasp_pose.p[0]:.3f}, {grasp_pose.p[1]:.3f}, {grasp_pose.p[2]:.3f}]")
    
    # -------------------------------------------------------------------------- #
    # Phase 1: Open gripper
    # -------------------------------------------------------------------------- #
    planner.open_gripper(robot_idx, t=3)
    
    # -------------------------------------------------------------------------- #
    # Phase 2: Move to above the cube (safe approach position)
    # -------------------------------------------------------------------------- #
    # Move to a position directly above the cube first
    # This avoids horizontal collision with the cube
    # Try progressively lower heights until one works
    approach_heights = [0.06, 0.05, 0.04, 0.03]  # 增加更多尝试选项
    result = -1
    successful_height = None  # Track which height worked for THIS robot
    
    for approach_height in approach_heights:
        above_cube_pose = sapien.Pose([0, 0, approach_height]) * grasp_pose
        result = planner.move_robot_to_pose(robot_idx, above_cube_pose)
        if result != -1:
            successful_height = approach_height
            # print(f"Robot {robot_idx + 1} successfully moved to {approach_height*100:.0f}cm above cube")
            break
        else:
            # print(f"Robot {robot_idx + 1} failed at {approach_height*100:.0f}cm above cube, trying lower...")
            pass
    if result == -1:
        # Try side approach instead of top-down
        print(f"Robot {robot_idx + 1} failed all top approaches, trying side approach...")
        
        # Determine side offset based on robot position
        # Robot 1 approaches from left (negative y), Robot 2 from right (positive y)
        side_offset = -0.08 if robot_idx == 0 else 0.08
        side_approach_pose = sapien.Pose([0, side_offset, 0]) * grasp_pose
        result = planner.move_robot_to_pose(robot_idx, side_approach_pose, refine_steps=8)
        
        if result != -1:
            # Successfully reached side position, now move to grasp
            print(f"Robot {robot_idx + 1} using side approach")
            result = planner.move_robot_to_pose(robot_idx, grasp_pose, refine_steps=10)
            if result == -1:
                print(f"Robot {robot_idx + 1} failed to move from side to grasp")
                return -1
            successful_height = 0
        else:
            # Last resort: try to go directly to grasp position
            print(f"Robot {robot_idx + 1} trying direct grasp as final attempt...")
            result = planner.move_robot_to_pose(robot_idx, grasp_pose, refine_steps=8)
            if result == -1:
                print(f"Robot {robot_idx + 1} completely failed to reach cube")
                return -1
            successful_height = 0  # Mark as direct grasp
    
            # Skip normal descent, already at grasp position
    else:
        # -------------------------------------------------------------------------- #
        # Phase 3: Descend to reach position (prepare for grasp)
        # -------------------------------------------------------------------------- #
        # Determine descent height based on current height
        current_height = successful_height if successful_height is not None else 0.04
        
        # Multi-waypoint descent for better success rate
        if current_height <= 0.05:
            # Already quite low, use fewer waypoints
            descent_waypoints = [0.03, 0.015, 0]
        else:
            # Higher position, use more waypoints for smoother descent
            descent_waypoints = [0.04, 0.03, 0.02, 0.01, 0]
        
        # Gradual descent through waypoints
        for waypoint_height in descent_waypoints:
            waypoint_pose = sapien.Pose([0, 0, waypoint_height]) * grasp_pose
            result = planner.move_robot_to_pose(robot_idx, waypoint_pose, refine_steps=4)
            if result == -1:
                # If intermediate waypoint fails, try to go directly to final grasp
                print(f"Robot {robot_idx + 1} failed at waypoint {waypoint_height*100:.1f}cm, trying direct descent...")
                result = planner.move_robot_to_pose(robot_idx, grasp_pose, refine_steps=8)
                if result == -1:
                    print(f"Robot {robot_idx + 1} failed direct descent to grasp position")
                    return -1
                break  # Successfully reached grasp, exit waypoint loop
    
    # -------------------------------------------------------------------------- #
    # Phase 5: Close gripper and stabilize
    # -------------------------------------------------------------------------- #
    # Close gripper to grasp - increase time for more stable grasp
    planner.close_gripper(robot_idx, t=25)  # 增加到25步，确保完全闭合
    
    # Wait a bit for the gripper to fully close and stabilize
    qpos1 = planner._get_current_qpos(0)[:len(planner.planner1.joint_vel_limits)]
    qpos2 = planner._get_current_qpos(1)[:len(planner.planner2.joint_vel_limits)]
    for _ in range(12):  # 增加稳定步数
        action1 = planner._make_action(qpos1, planner.gripper_state1)
        action2 = planner._make_action(qpos2, planner.gripper_state2)
        planner._step_env(action1, action2)
    
    # Verify grasp by checking cube position relative to gripper
    agent = planner.agent1 if robot_idx == 0 else planner.agent2
    cube_pos_after_grasp = cube.pose.sp.p
    tcp_pos_after_grasp = agent.tcp_pose.sp.p
    
    # Convert tensors to numpy if needed
    if hasattr(tcp_pos_after_grasp, 'cpu'):
        tcp_pos_after_grasp = tcp_pos_after_grasp.cpu().numpy()[0] if len(tcp_pos_after_grasp.shape) > 1 else tcp_pos_after_grasp.cpu().numpy()
    if hasattr(cube_pos_after_grasp, 'cpu'):
        cube_pos_after_grasp = cube_pos_after_grasp.cpu().numpy()[0] if len(cube_pos_after_grasp.shape) > 1 else cube_pos_after_grasp.cpu().numpy()
    
    grasp_distance = np.linalg.norm(cube_pos_after_grasp - tcp_pos_after_grasp)
    
    # If grasp seems failed (cube too far from gripper), try to regrasp
    if grasp_distance > 0.04:  # 4cm threshold
        print(f"Robot {robot_idx + 1} grasp verification failed (distance: {grasp_distance:.3f}m), attempting regrasp...")
        
        # Open gripper
        planner.open_gripper(robot_idx, t=10)
        
        # Move slightly up
        retreat_a_bit = sapien.Pose([0, 0, 0.02]) * grasp_pose
        planner.move_robot_to_pose(robot_idx, retreat_a_bit, refine_steps=3)
        
        # Try grasp again with adjusted position
        adjusted_grasp = compute_grasp_pose(env, cube, agent)
        result = planner.move_robot_to_pose(robot_idx, adjusted_grasp, refine_steps=8)
        if result == -1:
            print(f"Robot {robot_idx + 1} regrasp failed")
            return -1
        
        # Close gripper again
        planner.close_gripper(robot_idx, t=25)
        for _ in range(12):
            action1 = planner._make_action(qpos1, planner.gripper_state1)
            action2 = planner._make_action(qpos2, planner.gripper_state2)
            planner._step_env(action1, action2)
    
    # -------------------------------------------------------------------------- #
    # Phase 6: Lift - lift the cube up slowly and steadily
    # -------------------------------------------------------------------------- #
    # First a small lift to ensure cube is grasped
    small_lift_pose = sapien.Pose([0, 0, 0.02]) * grasp_pose
    result = planner.move_robot_to_pose(robot_idx, small_lift_pose, refine_steps=5)
    if result == -1:
        # print(f"Warning: Robot {robot_idx + 1} failed small lift, trying larger lift...")
        pass
    
    # Then lift higher for transport
    lift_pose = sapien.Pose([0, 0, 0.06]) * grasp_pose
    result = planner.move_robot_to_pose(robot_idx, lift_pose, refine_steps=3)
    if result == -1:
        # print(f"Warning: Robot {robot_idx + 1} failed to lift fully, continuing with partial lift...")
        # Continue even if lift fails
        pass
    
    # -------------------------------------------------------------------------- #
    # Phase 7: Move to goal - transport cube to target position
    # -------------------------------------------------------------------------- #
    # Use an elevated goal pose for safe transport
    elevated_target = target_pos.copy()
    elevated_target[2] += 0.05  # Lift by 5cm for safer transport
    goal_pose = sapien.Pose(elevated_target, grasp_pose.q)
    result = planner.move_robot_to_pose(robot_idx, goal_pose, refine_steps=5)  # Slower, smoother motion
    if result == -1:
        # print(f"Warning: Robot {robot_idx + 1} failed to reach elevated goal pose")
        # Try with less elevation
        elevated_target[2] = target_pos[2] + 0.03
        goal_pose = sapien.Pose(elevated_target, grasp_pose.q)
        result = planner.move_robot_to_pose(robot_idx, goal_pose, refine_steps=8)
        if result == -1:
            print(f"Warning: Robot {robot_idx + 1} failed to reach goal, trying direct placement...")
            return -1
    
    # -------------------------------------------------------------------------- #
    # Phase 8: Place - lower the cube gently and release
    # -------------------------------------------------------------------------- #
    # First, move to slightly above target for controlled descent
    pre_place_pose = sapien.Pose([target_pos[0], target_pos[1], target_pos[2] + 0.02], grasp_pose.q)
    result = planner.move_robot_to_pose(robot_idx, pre_place_pose, refine_steps=8)
    if result == -1:
        print(f"Warning: Robot {robot_idx + 1} failed to reach pre-place position...")
    
    # Now lower to final position slowly
    place_pose = sapien.Pose(target_pos, grasp_pose.q)
    result = planner.move_robot_to_pose(robot_idx, place_pose, refine_steps=12)  # 非常慢的下降
    if result == -1:
        print(f"Warning: Robot {robot_idx + 1} failed to place accurately, releasing anyway...")
    
    # Wait briefly before releasing to ensure cube is stable on surface
    qpos1 = planner._get_current_qpos(0)[:len(planner.planner1.joint_vel_limits)]
    qpos2 = planner._get_current_qpos(1)[:len(planner.planner2.joint_vel_limits)]
    for _ in range(10):  # 增加等待时间，确保稳定
        action1 = planner._make_action(qpos1, planner.gripper_state1)
        action2 = planner._make_action(qpos2, planner.gripper_state2)
        planner._step_env(action1, action2)
    
    # Open gripper to release slowly
    planner.open_gripper(robot_idx, t=15)  # 更慢的释放
    
    # Verify placement - check if cube is reasonably close to target
    cube_pos_after_place = cube.pose.sp.p
    if hasattr(cube_pos_after_place, 'cpu'):
        cube_pos_after_place = cube_pos_after_place.cpu().numpy()[0] if len(cube_pos_after_place.shape) > 1 else cube_pos_after_place.cpu().numpy()
    
    placement_error = np.linalg.norm(cube_pos_after_place[:2] - target_pos[:2])  # Check x,y only
    
    # If placement is way off (>5cm), try gentle push adjustment
    if placement_error > 0.05:
        print(f"Robot {robot_idx + 1} placement error {placement_error:.3f}m, attempting adjustment...")
        
        # Calculate push direction (from current TCP to target)
        agent = planner.agent1 if robot_idx == 0 else planner.agent2
        tcp_pos = agent.tcp_pose.sp.p
        if hasattr(tcp_pos, 'cpu'):
            tcp_pos = tcp_pos.cpu().numpy()[0] if len(tcp_pos.shape) > 1 else tcp_pos.cpu().numpy()
        
        # Push from slightly above the cube
        push_start = cube_pos_after_place.copy()
        push_start[2] += 0.03  # 3cm above cube
        push_pose = sapien.Pose(push_start, grasp_pose.q)
        
        # Move to push position
        result = planner.move_robot_to_pose(robot_idx, push_pose, refine_steps=5)
        if result != -1:
            # Gentle push towards target
            push_vector = target_pos[:2] - cube_pos_after_place[:2]
            push_vector = push_vector / np.linalg.norm(push_vector) * 0.02  # 2cm push
            
            push_target = push_start.copy()
            push_target[0] += push_vector[0]
            push_target[1] += push_vector[1]
            push_target_pose = sapien.Pose(push_target, grasp_pose.q)
            
            planner.move_robot_to_pose(robot_idx, push_target_pose, refine_steps=8)
    
    # -------------------------------------------------------------------------- #
    # Phase 9: Retreat - move back up slowly to avoid disturbing the cube
    # -------------------------------------------------------------------------- #
    # Wait a moment after releasing
    qpos1 = planner._get_current_qpos(0)[:len(planner.planner1.joint_vel_limits)]
    qpos2 = planner._get_current_qpos(1)[:len(planner.planner2.joint_vel_limits)]
    last_result = None
    for _ in range(8):  # 增加等待时间
        action1 = planner._make_action(qpos1, planner.gripper_state1)
        action2 = planner._make_action(qpos2, planner.gripper_state2)
        last_result = planner._step_env(action1, action2)
    
    # Retreat upward slowly
    retreat_pose = sapien.Pose([target_pos[0], target_pos[1], target_pos[2] + 0.10], grasp_pose.q)
    result = planner.move_robot_to_pose(robot_idx, retreat_pose, refine_steps=5)
    if result == -1:
        # print(f"Warning: Robot {robot_idx + 1} failed to retreat cleanly")
        # Return the last successful environment step result
        return last_result if last_result is not None else -1
    
    # Return the final environment step result (obs, reward, terminated, truncated, info)
    return result


def solve(env: SortCubeSO101Env, seed=None, debug=False, vis=False):
    """
    Motion planning solution for sorting cubes task.
    
    Robot 1 (index 0) handles RED cube only.
    Robot 2 (index 1) handles GREEN cube only.
    """
    env.reset(seed=seed)
    env_unwrapped = env.unwrapped
    
    # Create dual-arm motion planner
    planner = DualArmSO101MotionPlanner(
        env,
        debug=debug,
        vis=vis,
        print_env_info=False,
    )
    
    # Target positions
    red_target = np.array([
        env_unwrapped.red_cube_target_region_center[0],
        env_unwrapped.red_cube_target_region_center[1],
        env_unwrapped.cube_half_size + 0.02  # Slightly above table
    ])
    green_target = np.array([
        env_unwrapped.green_cube_target_region_center[0],
        env_unwrapped.green_cube_target_region_center[1],
        env_unwrapped.cube_half_size + 0.02
    ])
    
    # -------------------------------------------------------------------------- #
    # Robot 1: Pick and place RED cube
    # -------------------------------------------------------------------------- #
    print("=== Robot 1: Handling RED cube ===")
    result1 = pick_and_place(
        planner, 
        robot_idx=0, 
        cube=env_unwrapped.red_cube, 
        target_pos=red_target,
        env=env_unwrapped
    )
    
    # Check if Robot 1 failed
    if result1 == -1 or not isinstance(result1, tuple):
        print("Robot 1 failed to complete task")
        planner.close()
        return -1
    
    # -------------------------------------------------------------------------- #
    # Robot 2: Pick and place GREEN cube
    # -------------------------------------------------------------------------- #
    print("=== Robot 2: Handling GREEN cube ===")
    result2 = pick_and_place(
        planner, 
        robot_idx=1, 
        cube=env_unwrapped.green_cube, 
        target_pos=green_target,
        env=env_unwrapped
    )
    
    # Check if Robot 2 failed
    if result2 == -1 or not isinstance(result2, tuple):
        print("Robot 2 failed to complete task")
        planner.close()
        # Return result1 so we at least have Robot 1's successful actions recorded
        return result1
    
    planner.close()
    # Return the final result (contains obs, reward, terminated, truncated, info)
    # The info dict contains success status for the entire task (both cubes sorted)
    return result2
