import os
os.environ['MUJOCO_GL'] = 'egl'
import warnings
warnings.filterwarnings('ignore')

import hydra
import numpy as np
import torch
from termcolor import colored
from pathlib import Path

from common.parser import parse_cfg
from common.seed import set_seed
from tdmpc2 import TDMPC2
import gymnasium as gym
from functools import partial
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.utils.wrappers import FlattenRGBDObservationWrapper, FlattenActionSpaceWrapper
from gymnasium.spaces import Dict as DictSpace
from envs.wrappers.pixels import PixelWrapper

import mani_skill.envs
import grasp_cube.envs.tasks.pick_cube_so101
import grasp_cube.envs.tasks.lift_cube_so101
import grasp_cube.envs.tasks.stack_cube_so101
import grasp_cube.envs.tasks.sort_cube_so101

torch.backends.cudnn.benchmark = True


@hydra.main(config_name='config', config_path='.')
def collect_data(cfg: dict):
	"""
	Script for collecting trajectory data using a trained TD-MPC2 checkpoint.
	
	This script will:
	- Load a trained checkpoint
	- Run episodes in the environment
	- Save trajectories in ManiSkill's .h5 format
	- Optionally save videos
	
	Most relevant args:
		`env_id`: task name (eg. StackCubeSO101-v1)
		`model_size`: model size, must be one of `[1, 5, 19, 48, 317]`
		`checkpoint`: path to model checkpoint to load
		`num_episodes`: number of episodes to collect (default: 100)
		`output_dir`: directory to save trajectories (default: ./collected_data)
		`save_video`: whether to save videos (default: True)
		`seed`: random seed (default: 1)
	
	Example usage:
	```
		$ python collect_data.py checkpoint=logs/StackCubeSO101-v1/1/default/models/final.pt num_episodes=100 output_dir=./demo_data
	```
	"""
	assert torch.cuda.is_available()
	cfg = parse_cfg(cfg)
	set_seed(cfg.seed)
	
	# Get collection parameters
	num_episodes = cfg.get('num_episodes', 100)
	output_dir = cfg.get('output_dir', './collected_data')
	save_video = cfg.get('save_video', True)
	save_only_success = cfg.get('save_only_success', False)  # Note: filtering during collection is complex
	
	if save_only_success:
		print(colored('\n⚠️  Note: save_only_success is experimental. Consider setting it to False', 'yellow'))
		print(colored('    and filtering trajectories after collection for more reliable results.\n', 'yellow'))
	
	print(colored(f'=' * 80, 'cyan'))
	print(colored(f'Data Collection Configuration', 'cyan', attrs=['bold']))
	print(colored(f'=' * 80, 'cyan'))
	print(colored(f'Task: {cfg.env_id}', 'blue', attrs=['bold']))
	print(colored(f'Model size: {cfg.get("model_size", "default")}', 'blue', attrs=['bold']))
	print(colored(f'Checkpoint: {cfg.checkpoint}', 'blue', attrs=['bold']))
	print(colored(f'Number of episodes to collect: {num_episodes}', 'green', attrs=['bold']))
	print(colored(f'Output directory: {output_dir}', 'green', attrs=['bold']))
	print(colored(f'Save videos: {save_video}', 'green', attrs=['bold']))
	print(colored(f'Save only successful episodes: {save_only_success}', 'green', attrs=['bold']))
	print(colored(f'=' * 80, 'cyan'))
	
	# Create output directory
	output_path = Path(output_dir) / cfg.env_id
	output_path.mkdir(parents=True, exist_ok=True)
	
	# Make environment with RecordEpisode wrapper
	# Use a single environment for data collection to ensure proper recording
	env_make_fn = partial(
		gym.make,
		disable_env_checker=True,
		id=cfg.env_id,
		obs_mode=cfg.obs,
		render_mode='rgb_array',
		sensor_configs=dict(width=cfg.render_size, height=cfg.render_size),
		num_envs=1,  # Single environment for cleaner data collection
	)
	if cfg.control_mode != 'default':
		env_make_fn = partial(env_make_fn, control_mode=cfg.control_mode)
	
	# Create base environment (for GPU, this is already vectorized)
	env = env_make_fn()
	base_env = env.unwrapped  # Keep reference to base env
	
	# Get episode info
	max_episode_steps = env.spec.max_episode_steps if env.spec else 200
	
	# Flatten action space if needed
	if isinstance(env.action_space, DictSpace):
		env = FlattenActionSpaceWrapper(env)
	
	# Add RecordEpisode wrapper first to capture raw sensor data (both cameras)
	env = RecordEpisode(
		env,
		output_dir=str(output_path),
		trajectory_name="trajectory",
		max_steps_per_video=max_episode_steps,
		save_video=save_video,
		save_trajectory=True,
		save_on_reset=True,  # Auto save on reset
	)
	
	# Add observation wrappers for the agent
	# Note: We need to add a device attribute for PixelWrapper to work
	if cfg['obs'] == 'rgb':
		env = FlattenRGBDObservationWrapper(env, rgb=True, depth=False, state=cfg.include_state)
		# Add device attribute to env for PixelWrapper
		env.device = base_env.device
		env = PixelWrapper(cfg, env, num_envs=1)
	
	# Set config values needed by the agent (normally done by make_envs)
	try:  # Dict observation space
		cfg.obs_shape = {k: v.shape[1:] for k, v in env.observation_space.spaces.items()}
	except:  # Box observation space
		cfg.obs_shape = {cfg.get('obs', 'state'): env.observation_space.shape[1:]}
	cfg.action_dim = env.action_space.shape[-1]
	cfg.episode_length = max_episode_steps if max_episode_steps else 50
	cfg.seed_steps = max(1000, cfg.num_envs * cfg.episode_length)
	
	# Load agent
	agent = TDMPC2(cfg)
	assert os.path.exists(cfg.checkpoint), f'Checkpoint {cfg.checkpoint} not found!'
	agent.load(cfg.checkpoint)
	print(colored(f'\n✓ Agent loaded successfully from {cfg.checkpoint}\n', 'green', attrs=['bold']))
	
	# Collect data
	print(colored(f'Starting data collection...', 'yellow', attrs=['bold']))
	
	collected_episodes = 0
	total_success = 0
	total_reward = 0
	
	# Get device from base environment
	device = base_env.device
	
	for ep_idx in range(num_episodes):
		obs, _ = env.reset()
		done = torch.tensor([False], device=device)
		ep_reward = torch.tensor([0.0], device=device)
		t = 0
		episode_success = False
		
		# Run episode
		while not done[0]:
			action = agent.act(obs, t0=(t==0), eval_mode=True)
			obs, reward, terminated, truncated, info = env.step(action)
			done = terminated | truncated
			ep_reward += reward
			t += 1
		
		# Check if episode was successful
		if 'success' in info:
			if 'final_info' in info:
				episode_success = info['final_info']['success'][0].item()
			else:
				episode_success = info['success'][0].item() if hasattr(info['success'], '__getitem__') else info['success']
		
		# Record statistics (trajectory is saved automatically on next reset)
		collected_episodes += 1
		total_success += int(episode_success)
		total_reward += ep_reward.item()
		
		status = colored('SUCCESS ✅', 'green') if episode_success else colored('FAILURE ❌', 'red')
		print(colored(f'Episode {collected_episodes}/{num_episodes}', 'cyan') + 
			  f' | Status: {status} | ' +
			  f'Reward: {ep_reward.item():.2f} | ' +
			  f'Steps: {t}')
	
	# Final statistics
	print(colored(f'\n' + '=' * 80, 'cyan'))
	print(colored(f'Data Collection Complete!', 'green', attrs=['bold']))
	print(colored(f'=' * 80, 'cyan'))
	print(colored(f'Total episodes collected: {collected_episodes}', 'blue'))
	print(colored(f'Successful episodes: {total_success}/{collected_episodes} ({100*total_success/collected_episodes:.1f}%)', 'green', attrs=['bold']))
	print(colored(f'Failed episodes: {collected_episodes - total_success}/{collected_episodes}', 'blue'))
	print(colored(f'Average reward: {total_reward/collected_episodes:.2f}', 'blue'))
	print(colored(f'Data saved to: {output_path}', 'green', attrs=['bold']))
	
	if total_success < collected_episodes and save_only_success:
		print(colored(f'\n💡 Tip: You can filter out failed trajectories using:', 'yellow'))
		print(colored(f'   python filter_successful_trajectories.py {output_path}', 'yellow'))
	
	print(colored(f'=' * 80, 'cyan'))
	
	# Print trajectory file info
	h5_files = list(output_path.glob('trajectory.*.h5'))
	if h5_files:
		print(colored(f'\n✓ Trajectory files created:', 'green'))
		for f in sorted(h5_files)[-5:]:  # Show last 5 files
			print(f'  - {f.name}')
		if len(h5_files) > 5:
			print(f'  ... and {len(h5_files) - 5} more')
	
	env.close()


if __name__ == '__main__':
	collect_data()

