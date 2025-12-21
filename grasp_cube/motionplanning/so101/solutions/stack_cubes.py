import numpy as np
import sapien
from transforms3d.euler import euler2quat
from mani_skill.utils.structs.pose import to_sapien_pose

from grasp_cube.envs.tasks.stack_cube_so101 import StackCubeSO101Env
from grasp_cube.motionplanning.so101.motionplanner import SO101ArmMotionPlanningSolver
from grasp_cube.motionplanning.base_motionplanner.utils import (
    compute_grasp_info_by_obb, get_actor_obb)


def compute_grasp_pose(env: StackCubeSO101Env, cube):
    """Compute grasp pose for a cube."""
    FINGER_LENGTH = 0.025
    obb = get_actor_obb(cube)
    approaching = np.array([0, 0, -1])

    # Rotate around x-axis to align with the expected frame
    tcp_pose = sapien.Pose(q=euler2quat(np.pi / 2, 0, 0)) * env.agent.tcp_pose.sp
    target_closing = tcp_pose.to_transformation_matrix()[:3, 1]
    
    grasp_info = compute_grasp_info_by_obb(
        obb,
        approaching=approaching,
        target_closing=target_closing,
        depth=FINGER_LENGTH,
    )
    
    # Build grasp pose at cube position
    grasp_pose = env.agent.build_grasp_pose(approaching, grasp_info["closing"], cube.pose.sp.p)
    
    # Transform for SO101
    grasp_pose = grasp_pose * sapien.Pose(q=np.array([-1, 0, 0, 1]) / np.sqrt(2))
    
    return grasp_pose


def solve(env: StackCubeSO101Env, seed=None, debug=False, vis=False):
    """
    Motion planning solution for stacking cubes task.
    
    Task: Pick up the red cube and place it on top of the green cube.
    
    Args:
        env: The StackCubeSO101Env environment
        seed: Random seed for reset
        debug: Enable debug mode
        vis: Enable visualization
    
    Returns:
        The final step result (obs, reward, terminated, truncated, info)
    """
    env.reset(seed=seed)
    env_unwrapped = env.unwrapped
    
    # Create motion planner
    planner = SO101ArmMotionPlanningSolver(
        env,
        debug=debug,
        vis=vis,
        base_pose=to_sapien_pose(env_unwrapped.agent.robot.pose),
        visualize_target_grasp_pose=vis,
        print_env_info=False,
    )
    
    # Get cube parameters
    cube_half_size = env_unwrapped.cube_half_size
    
    # -------------------------------------------------------------------------- #
    # Phase 1: Open gripper
    # -------------------------------------------------------------------------- #
    planner.open_gripper()
    
    # -------------------------------------------------------------------------- #
    # Phase 2: Compute grasp pose for red cube
    # -------------------------------------------------------------------------- #
    grasp_pose = compute_grasp_pose(env_unwrapped, env_unwrapped.red_cube)
    
    # -------------------------------------------------------------------------- #
    # Phase 3: Move to approach position (above the red cube)
    # -------------------------------------------------------------------------- #
    approach_pose = sapien.Pose([0, 0, 0.05]) * grasp_pose
    result = planner.move_to_pose_with_RRTConnect(approach_pose)
    if result == -1:
        print("Failed to move to approach position")
        planner.close()
        return -1
    
    # -------------------------------------------------------------------------- #
    # Phase 4: Move down to grasp position
    # -------------------------------------------------------------------------- #
    result = planner.move_to_pose_with_RRTConnect(grasp_pose)
    if result == -1:
        print("Failed to move to grasp position")
        planner.close()
        return -1
    
    # -------------------------------------------------------------------------- #
    # Phase 5: Close gripper to grasp the red cube
    # -------------------------------------------------------------------------- #
    planner.close_gripper(t=15)
    
    # -------------------------------------------------------------------------- #
    # Phase 6: Lift the red cube
    # -------------------------------------------------------------------------- #
    lift_pose = sapien.Pose([0, 0, 0.08]) * grasp_pose
    result = planner.move_to_pose_with_RRTConnect(lift_pose)
    if result == -1:
        print("Failed to lift cube")
        # Try smaller lift
        lift_pose = sapien.Pose([0, 0, 0.05]) * grasp_pose
        result = planner.move_to_pose_with_RRTConnect(lift_pose)
    
    # -------------------------------------------------------------------------- #
    # Phase 7: Calculate target position (on top of green cube)
    # -------------------------------------------------------------------------- #
    # The target is on top of the green cube
    # Target z = green_cube.z + cube_half_size (green cube top) + cube_half_size (red cube bottom to center)
    green_cube_pos = env_unwrapped.green_cube.pose.sp.p
    
    # Convert to numpy if needed
    if hasattr(green_cube_pos, 'cpu'):
        green_cube_pos = green_cube_pos.cpu().numpy()
        if len(green_cube_pos.shape) > 1:
            green_cube_pos = green_cube_pos[0]
    
    target_pos = np.array([
        green_cube_pos[0],
        green_cube_pos[1],
        green_cube_pos[2] + 2 * cube_half_size + 0.01  # Slightly above to account for placement
    ])
    
    # -------------------------------------------------------------------------- #
    # Phase 8: Move above the green cube
    # -------------------------------------------------------------------------- #
    # First move to a position above the target
    above_target_pose = sapien.Pose(
        [target_pos[0], target_pos[1], target_pos[2] + 0.05],
        grasp_pose.q
    )
    result = planner.move_to_pose_with_RRTConnect(above_target_pose)
    if result == -1:
        print("Failed to move above green cube")
        # Try alternative path
        above_target_pose = sapien.Pose(
            [target_pos[0], target_pos[1], target_pos[2] + 0.08],
            grasp_pose.q
        )
        result = planner.move_to_pose_with_RRTConnect(above_target_pose)
    
    # -------------------------------------------------------------------------- #
    # Phase 9: Lower onto the green cube
    # -------------------------------------------------------------------------- #
    place_pose = sapien.Pose(target_pos, grasp_pose.q)
    result = planner.move_to_pose_with_RRTConnect(place_pose)
    if result == -1:
        print("Failed to lower onto green cube, trying alternative approach")
        # Try slightly higher position
        place_pose = sapien.Pose(
            [target_pos[0], target_pos[1], target_pos[2] + 0.01],
            grasp_pose.q
        )
        result = planner.move_to_pose_with_RRTConnect(place_pose)
    
    # -------------------------------------------------------------------------- #
    # Phase 10: Open gripper to release the red cube
    # -------------------------------------------------------------------------- #
    planner.open_gripper(t=10)
    
    # -------------------------------------------------------------------------- #
    # Phase 11: Retreat upward
    # -------------------------------------------------------------------------- #
    retreat_pose = sapien.Pose(
        [target_pos[0], target_pos[1], target_pos[2] + 0.08],
        grasp_pose.q
    )
    result = planner.move_to_pose_with_RRTConnect(retreat_pose)
    
    # Wait for things to settle
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(np.zeros(env.action_space.shape[-1]))
        if vis:
            env_unwrapped.render_human()
    
    planner.close()
    
    # Return the final result
    return obs, reward, terminated, truncated, info

