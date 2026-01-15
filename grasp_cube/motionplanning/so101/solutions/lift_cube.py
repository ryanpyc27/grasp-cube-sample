import numpy as np
import sapien
from transforms3d.euler import euler2quat

from mani_skill.agents.base_agent import BaseAgent
from grasp_cube.envs.tasks.lift_cube_so101 import LiftCubeSO101Env
from grasp_cube.motionplanning.so101.motionplanner import SO101ArmMotionPlanningSolver
from grasp_cube.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


def compute_grasp_pose(env: LiftCubeSO101Env, cube, agent: BaseAgent) -> sapien.Pose:
    """Compute grasp pose for a cube using specified agent.
    Borrowed from self_defined.py"""
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
    
    # Build grasp pose at cube position
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


def solve(env: LiftCubeSO101Env, seed=None, debug=False, vis=False):
    """
    Motion planning solution for lifting cube task.
    
    Task: Pick up cube and lift it 6cm (0.06m) upward
    """
    env.reset(seed=seed)
    env_unwrapped = env.unwrapped
    
    # Create motion planner
    planner = SO101ArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=env_unwrapped.agent.robot.pose,
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    
    agent = env_unwrapped.agent
    cube = env_unwrapped.cube
    
    # -------------------------------------------------------------------------- #
    # Phase 1: Open gripper
    # -------------------------------------------------------------------------- #
    print("Phase 1: Opening gripper...")
    planner.open_gripper()
    
    # -------------------------------------------------------------------------- #
    # Phase 2: Compute grasp pose
    # -------------------------------------------------------------------------- #
    print("Phase 2: Computing grasp pose...")
    grasp_pose = compute_grasp_pose(env_unwrapped, cube, agent)
    
    # -------------------------------------------------------------------------- #
    # Phase 3: Move to approach position (reach)
    # -------------------------------------------------------------------------- #
    print("Phase 3: Moving to approach position...")
    reach_pose = sapien.Pose([0, 0.02, 0.03]) * grasp_pose
    planner.move_to_pose_with_RRTConnect(reach_pose)
    
    # -------------------------------------------------------------------------- #
    # Phase 4: Move to grasp position
    # -------------------------------------------------------------------------- #
    print("Phase 4: Moving to grasp position...")
    planner.move_to_pose_with_RRTConnect(sapien.Pose([0, 0, 0]) * grasp_pose)
    
    # -------------------------------------------------------------------------- #
    # Phase 5: Close gripper to grasp cube
    # -------------------------------------------------------------------------- #
    print("Phase 5: Grasping cube...")
    planner.close_gripper(t=12)
    
    # -------------------------------------------------------------------------- #
    # Phase 6: Move to upright pose to lift cube
    # -------------------------------------------------------------------------- #
    print("Phase 6: Lifting cube 6cm upward...")
    # Get current TCP pose
    current_tcp_pose = agent.tcp_pose.sp
    
    # Create lifted pose: same orientation, but 6cm higher in z-axis
    lift_height = 0.05  # 6cm
    lifted_pose = sapien.Pose(
        p=current_tcp_pose.p + np.array([0, 0, lift_height]),
        q=grasp_pose.q  # Keep same orientation as grasp
    )
    
    # Move to lifted position
    res = planner.move_to_pose_with_RRTConnect(lifted_pose)
    
    print("Task completed: Cube lifted!")
    
    planner.close()
    return res
