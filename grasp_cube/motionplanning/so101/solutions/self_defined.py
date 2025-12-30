import numpy as np
import sapien
import mplib
from transforms3d.euler import euler2quat
from transforms3d import euler
from transforms3d.quaternions import qmult

from mani_skill.agents.base_agent import BaseAgent
from mani_skill.utils.structs.pose import to_sapien_pose
from grasp_cube.envs.tasks.self_defined_so101 import SelfDefinedSO101Env
from grasp_cube.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


class DualArmSO101MotionPlanner:
    """
    Dual-arm motion planner for SO101 robots.
    Handles two robots independently while stepping the environment with combined actions.
    """
    OPEN = 0.6
    CLOSED = 0.0
    MOVE_GROUP = "gripper_link_tip"
    
    def __init__(
        self,
        env: SelfDefinedSO101Env,
        debug: bool = False,
        vis: bool = True,
        print_env_info: bool = False,
        joint_vel_limits: float = 0.6,
        joint_acc_limits: float = 0.6,
    ):
        self.env = env
        self.base_env: SelfDefinedSO101Env = env.unwrapped
        self.debug = debug
        self.vis = vis
        self.print_env_info = print_env_info
        self.elapsed_steps = 0
        self.joint_vel_limits = joint_vel_limits
        self.joint_acc_limits = joint_acc_limits
        
        # Get agents (robots)
        self.agent1: BaseAgent = self.base_env.agent.agents[0]  # Robot 1 - opens/closes drawer
        self.agent2: BaseAgent = self.base_env.agent.agents[1]  # Robot 2 - handles red cube
        
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
    
    def close_gripper(self, robot_idx: int, t: int = 6, gripper_state: float = None):
        """Close gripper for specified robot (0 or 1)."""
        if gripper_state is None:
            gripper_state = self.CLOSED
        
        if robot_idx == 0:
            self.gripper_state1 = gripper_state
        else:
            self.gripper_state2 = gripper_state
        
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
            planning_time=10.0,
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


def get_drawer_handle_pose(env: SelfDefinedSO101Env, agent: BaseAgent, drawer_idx: int = 0) -> sapien.Pose:
    """
    Get the pose of the drawer handle for grasping.
    
    Args:
        env: The environment
        agent: The agent (robot) that will grasp the handle
        drawer_idx: Index of the drawer (0 = bottom drawer)
    
    Returns:
        Pose of the drawer handle
    """
    # Use current TCP pose as reference to ensure reachability
    current_tcp_pose = agent.tcp_pose.sp
    current_tcp_p = current_tcp_pose.p
    current_tcp_q = current_tcp_pose.q
    
    # Convert to numpy if needed
    if hasattr(current_tcp_p, 'cpu'):
        current_tcp_p = current_tcp_p.cpu().numpy()
    if hasattr(current_tcp_q, 'cpu'):
        current_tcp_q = current_tcp_q.cpu().numpy()
    if len(current_tcp_p.shape) > 1:
        current_tcp_p = current_tcp_p[0]
    if len(current_tcp_q.shape) > 1:
        current_tcp_q = current_tcp_q[0]
    
    # Set handle position very close to current TCP - just move slightly in +y direction
    # Current TCP: [0.301, 0.275, 0.091]
    # Move just 2cm toward cabinet (+y direction)
    handle_pos = [0.301, 0.360, 0.300]
    
    # Rotate current TCP orientation by 90 degrees
    # Choose which axis to rotate around:
    
    # Option 1: Rotate 90° around x-axis
    # rotation_quat = euler2quat(np.pi / 2, 0, 0)  # -90° around x-axis
    # handle_quat = qmult(current_tcp_q, rotation_quat)
    
    # Option 2: Rotate 90° around y-axis
    rotation_quat = euler2quat(0, np.pi/2, 0)  # 90° around y-axis
    handle_quat = qmult(current_tcp_q, rotation_quat)
    
    # Option 3: Rotate 90° around z-axis
    # rotation_quat = euler2quat(0, 0, np.pi/2)  # 90° around z-axis
    # handle_quat = qmult(current_tcp_q, rotation_quat)
    
    return sapien.Pose(p=handle_pos, q=handle_quat)


def open_drawer(planner: DualArmSO101MotionPlanner, robot_idx: int, 
                drawer_idx: int, open_amount: float = 0.12):
    """
    Open a drawer by pulling it with the robot.
    
    Args:
        planner: Motion planner
        robot_idx: Robot to use (typically 0 for robot1)
        drawer_idx: Which drawer to open (0 = bottom)
        open_amount: How much to open in meters
    """
    print(f"=== Robot {robot_idx + 1}: Opening drawer {drawer_idx} ===")
    env = planner.base_env
    agent = planner.agent1 if robot_idx == 0 else planner.agent2
    robot = agent.robot
    
    # Debug: print robot info
    robot_pos = robot.pose.p
    tcp_pos = agent.tcp_pose.sp.p
    if hasattr(robot_pos, 'cpu'):
        robot_pos = robot_pos.cpu().numpy()[0]
    if hasattr(tcp_pos, 'cpu'):
        tcp_pos = tcp_pos.cpu().numpy()[0] if len(tcp_pos.shape) > 1 else tcp_pos.cpu().numpy()
    print(f"Robot {robot_idx + 1} base at: [{robot_pos[0]:.3f}, {robot_pos[1]:.3f}, {robot_pos[2]:.3f}]")
    print(f"Robot {robot_idx + 1} TCP at: [{tcp_pos[0]:.3f}, {tcp_pos[1]:.3f}, {tcp_pos[2]:.3f}]")
    
    # Get drawer handle pose
    handle_pose = get_drawer_handle_pose(env, agent, drawer_idx)
    print(f"Drawer handle at: [{handle_pose.p[0]:.3f}, {handle_pose.p[1]:.3f}, {handle_pose.p[2]:.3f}]")
    print(f"Handle quaternion: [{handle_pose.q[0]:.3f}, {handle_pose.q[1]:.3f}, {handle_pose.q[2]:.3f}, {handle_pose.q[3]:.3f}]")
    
    # Calculate distance
    dist = np.linalg.norm(handle_pose.p - robot_pos)
    print(f"Distance from robot base to handle: {dist:.3f}m")
    
    # Phase 1: Open gripper and move to approach position
    planner.open_gripper(robot_idx, t=6)
    
    # Get current TCP position for debugging
    current_tcp = agent.tcp_pose.sp.p
    if hasattr(current_tcp, 'cpu'):
        current_tcp = current_tcp.cpu().numpy()
    if len(current_tcp.shape) > 1:
        current_tcp = current_tcp[0]
    
    target_pos = handle_pose.p.copy()
    
    print(f"Current TCP: [{current_tcp[0]:.3f}, {current_tcp[1]:.3f}, {current_tcp[2]:.3f}]")
    print(f"Target handle: [{target_pos[0]:.3f}, {target_pos[1]:.3f}, {target_pos[2]:.3f}]")
    
    # # Move to approach position (slightly back from handle in +y direction)
    approach_pos = target_pos.copy()
    approach_pos[1] -= 0.06
    approach_pose = sapien.Pose(p=approach_pos, q=handle_pose.q)
    print(f"Approach pose at: [{approach_pos[0]:.3f}, {approach_pos[1]:.3f}, {approach_pos[2]:.3f}]")
    result = planner.move_robot_to_pose(robot_idx, approach_pose, refine_steps=5)
    if result == -1:
        print(f"Robot {robot_idx + 1} failed to approach drawer, trying direct grasp...")
        # If approach fails, try to go directly to handle
        # Don't return -1 yet, continue to phase 2
    
    # Phase 2: Move to handle and grasp
    result = planner.move_robot_to_pose(robot_idx, handle_pose, refine_steps=8)
    if result == -1:
        print(f"Robot {robot_idx + 1} failed to reach handle")
        return -1
    
    # Close gripper on handle - use longer time to grasp firmly
    print(f"Grasping handle firmly...")
    planner.close_gripper(robot_idx, t=30)  # Increased from 20 to 30
    
    # Stabilize grasp - much longer to ensure firm grip before pulling
    qpos1 = planner._get_current_qpos(0)[:len(planner.planner1.joint_vel_limits)]
    qpos2 = planner._get_current_qpos(1)[:len(planner.planner2.joint_vel_limits)]
    for _ in range(30):  # Increased from 10 to 30
        action1 = planner._make_action(qpos1, planner.gripper_state1)
        action2 = planner._make_action(qpos2, planner.gripper_state2)
        planner._step_env(action1, action2)
    
    print(f"Grip stabilized, ready to pull")
    
    # Phase 3: Pull drawer open in small increments
    # Pull in small steps to avoid IK failures
    print(f"Pulling drawer open in small steps...")
    num_pull_steps = 1  # More steps = smaller movements
    step_size = open_amount / num_pull_steps  # Each step is ~0.01m
    
    for step in range(num_pull_steps):
        # Calculate target position for this step
        pull_pos = handle_pose.p.copy()
        pull_pos[1] -= step_size * (step + 1)  # Incrementally pull in -y direction
        
        # Get current TCP orientation (allow it to change naturally)
        current_tcp_pose = agent.tcp_pose.sp
        current_tcp_q = current_tcp_pose.q
        if hasattr(current_tcp_q, 'cpu'):
            current_tcp_q = current_tcp_q.cpu().numpy()
        if len(current_tcp_q.shape) > 1:
            current_tcp_q = current_tcp_q[0]
        
        pull_pose = sapien.Pose(p=pull_pos, q=current_tcp_q)  # Use current orientation
        
        print(f"Pull step {step+1}/{num_pull_steps}: moving to [{pull_pos[0]:.3f}, {pull_pos[1]:.3f}, {pull_pos[2]:.3f}]")
        result = planner.move_robot_to_pose(robot_idx, pull_pose, refine_steps=10)
        
        if result == -1:
            print(f"Pull step {step+1} failed, drawer partially open")
            if step == 0:
                # If first step fails, this is a problem
                return -1
            else:
                # If later steps fail, we have at least partial opening
                break
        
        # Update drawer joint drive target to current position to prevent spring-back
        current_drawer_qpos = env.cabinet.get_qpos()
        if hasattr(current_drawer_qpos, 'cpu'):
            current_drawer_qpos = current_drawer_qpos.cpu().numpy()
        if len(current_drawer_qpos.shape) > 1:
            current_drawer_qpos = current_drawer_qpos[0]  # Get first batch
        drawer_pos = float(current_drawer_qpos[drawer_idx])
        env.drawer_joints[drawer_idx].set_drive_target(drawer_pos)
        
        # Longer stabilization after each step to let drawer settle
        # AND continuously update drive target to lock drawer position
        qpos1 = planner._get_current_qpos(0)[:len(planner.planner1.joint_vel_limits)]
        qpos2 = planner._get_current_qpos(1)[:len(planner.planner2.joint_vel_limits)]
    
    # After pulling is complete, lock the drawer at the open position
    # Increase stiffness and damping to prevent sliding back
    final_drawer_qpos = env.cabinet.get_qpos()
    if hasattr(final_drawer_qpos, 'cpu'):
        final_drawer_qpos = final_drawer_qpos.cpu().numpy()
    if len(final_drawer_qpos.shape) > 1:
        final_drawer_qpos = final_drawer_qpos[0]
    final_drawer_pos = float(final_drawer_qpos[drawer_idx])
    
    # Set stiffness to lock position, with moderate damping for stability
    env.drawer_joints[drawer_idx].set_drive_properties(stiffness=800, damping=80)
    env.drawer_joints[drawer_idx].set_drive_target(final_drawer_pos)
    print(f"Drawer locked at position: {final_drawer_pos:.4f} (stiffness=800, damping=80)")
    
    print(f"Robot {robot_idx + 1} successfully opened drawer (holding handle)")
    return 0  # Success


def close_drawer(planner: DualArmSO101MotionPlanner, robot_idx: int, drawer_idx: int):
    """
    Close a drawer by pushing it with the robot.
    
    Args:
        planner: Motion planner
        robot_idx: Robot to use (typically 0 for robot1)
        drawer_idx: Which drawer to close (0 = bottom)
    """
    print(f"=== Robot {robot_idx + 1}: Closing drawer {drawer_idx} ===")
    env = planner.base_env
    agent = planner.agent1 if robot_idx == 0 else planner.agent2
    
    # Push drawer closed in small increments
    print(f"Pushing drawer closed in small steps...")
    num_push_steps = 1
    step_size = 0.18 / num_push_steps  # Total push distance 0.08m
    
    # Get current TCP position
    current_tcp_pose = agent.tcp_pose.sp
    start_pos = current_tcp_pose.p
    if hasattr(start_pos, 'cpu'):
        start_pos = start_pos.cpu().numpy()
    if len(start_pos.shape) > 1:
        start_pos = start_pos[0]
    
    for step in range(num_push_steps):
        # Calculate target position for this step (push in +y direction)
        push_pos = start_pos.copy()
        push_pos[1] += step_size * (step + 1)  # Incrementally push in +y direction
        
        # Get current TCP orientation
        current_tcp_q = agent.tcp_pose.sp.q
        if hasattr(current_tcp_q, 'cpu'):
            current_tcp_q = current_tcp_q.cpu().numpy()
        if len(current_tcp_q.shape) > 1:
            current_tcp_q = current_tcp_q[0]
        
        push_pose = sapien.Pose(p=push_pos, q=current_tcp_q)
        
        print(f"Push step {step+1}/{num_push_steps}: moving to [{push_pos[0]:.3f}, {push_pos[1]:.3f}, {push_pos[2]:.3f}]")
        result = planner.move_robot_to_pose(robot_idx, push_pose, refine_steps=8)
        
        if result == -1:
            print(f"Push step {step+1} failed")
            if step == 0:
                print(f"Robot {robot_idx + 1} failed to close drawer")
                break
        
        # Small stabilization after each step
        qpos1 = planner._get_current_qpos(0)[:len(planner.planner1.joint_vel_limits)]
        qpos2 = planner._get_current_qpos(1)[:len(planner.planner2.joint_vel_limits)]
        for _ in range(5):
            action1 = planner._make_action(qpos1, planner.gripper_state1)
            action2 = planner._make_action(qpos2, planner.gripper_state2)
            planner._step_env(action1, action2)
    
    # Release gripper
    planner.open_gripper(robot_idx, t=10)
    
    print(f"Robot {robot_idx + 1} closed drawer")
    return 0  # Success


def compute_grasp_pose(env: SelfDefinedSO101Env, cube, agent: BaseAgent) -> sapien.Pose:
    """Compute grasp pose for a cube using specified agent.
    Mimics the implementation from pick_cube.py"""
    FINGER_LENGTH = 0.025
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
    
    # Build grasp pose at cube position (use actual cube position like pick_cube.py)
    cube_pos = cube.pose.sp.p
    # Convert to numpy if it's a tensor
    if hasattr(cube_pos, 'cpu'):
        cube_pos = cube_pos.cpu().numpy()
    if len(cube_pos.shape) > 1:
        cube_pos = cube_pos[0]
    
    grasp_pose = agent.build_grasp_pose(approaching, grasp_info["closing"], cube_pos)
    
    # Transform for SO101 (same as pick_cube.py)
    grasp_pose = grasp_pose * sapien.Pose(q=np.array([-1, 0, 0, 1]) / np.sqrt(2))
    
    return grasp_pose


def pick_and_place_in_drawer(planner: DualArmSO101MotionPlanner, robot_idx: int, 
                              cube, env: SelfDefinedSO101Env):
    """
    Pick up cube and place it into the open drawer.
    Exactly mimics pick_cube.py grasp approach.
    
    Args:
        planner: Motion planner
        robot_idx: Robot to use (typically 1 for robot2)
        cube: The cube to pick up
        env: Environment
    """
    print(f"=== Robot {robot_idx + 1}: Picking cube and placing in drawer ===")
    
    agent = planner.agent1 if robot_idx == 0 else planner.agent2
    
    # -------------------------------------------------------------------------- #
    # Compute grasp pose (exactly like pick_cube.py)
    # -------------------------------------------------------------------------- #
    grasp_pose = compute_grasp_pose(env, cube, agent)
    
    # -------------------------------------------------------------------------- #
    # Phase 1: Open gripper (exactly like pick_cube.py)
    # -------------------------------------------------------------------------- #
    planner.open_gripper(robot_idx, t=6)
    
    # -------------------------------------------------------------------------- #
    # Phase 2: Reach - move to approach position (exactly like pick_cube.py)
    # pick_cube.py line 50: reach_pose = sapien.Pose([0, 0.02, 0.03]) * grasp_pose
    # -------------------------------------------------------------------------- #
    reach_pose = sapien.Pose([0, -0.02, 0.01]) * grasp_pose
    result = planner.move_robot_to_pose(robot_idx, reach_pose)
    if result == -1:
        print(f"Robot {robot_idx + 1} failed to reach approach position")
        return -1
    
    # -------------------------------------------------------------------------- #
    # Phase 3: Grasp - descend to grasp position (exactly like pick_cube.py)
    # pick_cube.py line 59: planner.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0]) * grasp_pose)
    # -------------------------------------------------------------------------- #
    result = planner.move_robot_to_pose(robot_idx, sapien.Pose([0, 0, 0]) * grasp_pose, refine_steps=8)
    if result == -1:
        print(f"Robot {robot_idx + 1} failed to reach grasp position")
        return -1
    
    # -------------------------------------------------------------------------- #
    # Phase 4: Close gripper (exactly like pick_cube.py)
    # pick_cube.py line 60: planner.close_gripper(t=12)
    # -------------------------------------------------------------------------- #
    planner.close_gripper(robot_idx, t=12)
    
    print(f"Robot {robot_idx + 1} successfully grasped cube!")
    
    # Step 5b: Move to upright pose (end effector pointing up)
    print(f"\n--- Step 5b: Moving to upright position ---")
    
    # Create upright orientation (gripper pointing up: z-axis up, x-axis forward)
    # For SO101, upright means the gripper fingers point up
    upright_qpos = np.array([0.025150669738650322,
                            -0.33991164565086365,
                            -0.5317384004592896,
                            -0.30,
                            -1.8569945096969604,
                            0.031005658209323883
                            ])
    
    # 获取正确的robot的当前关节位置
    current_qpos = planner._get_current_qpos(robot_idx)
    planner_obj = planner.planner1 if robot_idx == 0 else planner.planner2
    joint_limits_len = len(planner_obj.joint_vel_limits)
    
    upright_qpos_trimmed = upright_qpos[:joint_limits_len]
    current_qpos_trimmed = current_qpos[:joint_limits_len]
    
    result_robot_upright = planner_obj.plan_qpos_to_qpos(
        [upright_qpos_trimmed],
        current_qpos_trimmed,
        time_step=planner.base_env.control_timestep,
        use_point_cloud=False,
        planning_time=10.0,
    )
    
    # 检查规划是否成功并执行
    if result_robot_upright["status"] == "Success":
        result = planner._follow_path(robot_idx, result_robot_upright, refine_steps=5)
        print(f"  ✓ Robot {robot_idx + 1} reached upright pose")
    else:
        print(f"  ⚠ Warning: Robot {robot_idx + 1} failed to plan upright ({result_robot_upright['status']})")
        # Continue anyway
    
    # Step 5c: Rotate base joint (joint 0) 45 degrees clockwise
    print(f"\n--- Step 5c: Rotating base joint 45° clockwise ---")
    current_qpos = planner._get_current_qpos(robot_idx)
    planner_obj = planner.planner1 if robot_idx == 0 else planner.planner2
    joint_limits_len = len(planner_obj.joint_vel_limits)
    current_qpos_trimmed = current_qpos[:joint_limits_len]
    
    # Only rotate joint 0 (base joint) by -45 degrees (clockwise)
    rotated_qpos = current_qpos_trimmed.copy()
    rotated_qpos[0] = current_qpos_trimmed[0] + np.pi * 8.0/ 36.0  # -45° = 顺时针
    
    result_rotate = planner_obj.plan_qpos_to_qpos(
        [rotated_qpos],
        current_qpos_trimmed,
        time_step=planner.base_env.control_timestep,
        use_point_cloud=False,
        planning_time=10.0,
    )
    
    if result_rotate["status"] == "Success":
        result = planner._follow_path(robot_idx, result_rotate, refine_steps=5)
        print(f"  ✓ Base joint rotated 45° clockwise")
    else:
        print(f"  ⚠ Warning: Failed to rotate base joint ({result_rotate['status']}), continuing anyway...")
    
    # Get updated orientation for later use
    agent = planner.agent1 if robot_idx == 0 else planner.agent2
    rotated_tcp_q = agent.tcp_pose.sp.q
    if hasattr(rotated_tcp_q, 'cpu'):
        rotated_tcp_q = rotated_tcp_q.cpu().numpy()
    if len(rotated_tcp_q.shape) > 1:
        rotated_tcp_q = rotated_tcp_q[0]
    rotated_quat = rotated_tcp_q
    if result == -1:
        print(f"Robot {robot_idx + 1} failed to reach above drawer after 40 attempts")
        print(f"  → Will try to adjust gripper and release from current position")
    
    
    # -------------------------------------------------------------------------- #
    # Phase 9: Release cube
    # -------------------------------------------------------------------------- #
    print(f"\n--- Phase 9: Releasing cube ---")
    planner.open_gripper(robot_idx, t=15)
    
    # Wait for cube to settle
    qpos1 = planner._get_current_qpos(0)[:len(planner.planner1.joint_vel_limits)]
    qpos2 = planner._get_current_qpos(1)[:len(planner.planner2.joint_vel_limits)]
    for _ in range(20):
        action1 = planner._make_action(qpos1, planner.gripper_state1)
        action2 = planner._make_action(qpos2, planner.gripper_state2)
        planner._step_env(action1, action2)
    
    print(f"Robot {robot_idx + 1} finished placing cube in drawer")
    return result


def solve(env: SelfDefinedSO101Env, seed=None, debug=False, vis=False):
    """
    Motion planning solution for cabinet task.
    
    Task: Put red cube into cabinet drawer
    Strategy:
    1. Robot 1 opens the bottom drawer
    2. Robot 2 picks up red cube and places it in the drawer
    3. Robot 1 closes the drawer
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
    
    # -------------------------------------------------------------------------- #
    # Phase 1: Robot 1 opens the bottom drawer
    # -------------------------------------------------------------------------- #
    print("\n" + "="*60)
    print("PHASE 1: Opening drawer")
    print("="*60)
    result = open_drawer(planner, robot_idx=0, drawer_idx=0, open_amount=0.18)
    if result == -1:
        print("Failed to open drawer")
        planner.close()
        return -1
    
    # Release the handle and move Robot1 away to avoid interference
    print("Robot 1: Releasing handle and moving away...")
    planner.open_gripper(robot_idx=0, t=20)  # Open gripper to release handle
    
    # Move Robot1 back to a safe position away from the drawer
    agent1 = planner.agent1
    current_tcp = agent1.tcp_pose.sp
    retreat_pose = sapien.Pose(p=current_tcp.p + np.array([0, -0.15, 0.05]), q=current_tcp.q)
    result = planner.move_robot_to_pose(0, retreat_pose, refine_steps=5)
    if result == -1:
        print("Warning: Robot 1 failed to retreat, continuing anyway...")
    
    # Stabilize after retreat
    qpos1 = planner._get_current_qpos(0)[:len(planner.planner1.joint_vel_limits)]
    qpos2 = planner._get_current_qpos(1)[:len(planner.planner2.joint_vel_limits)]
    for _ in range(20):
        action1 = planner._make_action(qpos1, planner.gripper_state1)
        action2 = planner._make_action(qpos2, planner.gripper_state2)
        planner._step_env(action1, action2)
    
    print("Robot 1 moved away, drawer should remain open")
    
    # -------------------------------------------------------------------------- #
    # Phase 2: Robot 2 picks red cube and places in drawer
    # -------------------------------------------------------------------------- #
    print("\n" + "="*60)
    print("PHASE 2: Placing cube in drawer")
    print("="*60)
    result = pick_and_place_in_drawer(
        planner, 
        robot_idx=1, 
        cube=env_unwrapped.red_cube,
        env=env_unwrapped
    )
    if result == -1:
        print("Failed to place cube in drawer")
        # Continue to close drawer anyway
    
    # # -------------------------------------------------------------------------- #
    # # Phase 3: Robot 1 closes the drawer
    # # -------------------------------------------------------------------------- #
    print("\n" + "="*60)
    print("PHASE 3: Closing drawer")
    print("="*60)
    result = close_drawer(planner, robot_idx=0, drawer_idx=0)
    if result == -1:
        print("Failed to close drawer completely")
        # Continue anyway
    
    # Final stabilization
    qpos1 = planner._get_current_qpos(0)[:len(planner.planner1.joint_vel_limits)]
    qpos2 = planner._get_current_qpos(1)[:len(planner.planner2.joint_vel_limits)]
    for _ in range(20):
        action1 = planner._make_action(qpos1, planner.gripper_state1)
        action2 = planner._make_action(qpos2, planner.gripper_state2)
        last_result = planner._step_env(action1, action2)
    
    planner.close()
    
    print("\n" + "="*60)
    print("Task completed!")
    print("="*60)
    
    # Return the final result
    return last_result
