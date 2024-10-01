import argparse
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from collections import deque
import pickle
import h5py

import TD7


def train_offline(RL_agent, args):
	# Load the custom dataset and perfect trajectories
	RL_agent.replay_buffer.load_custom_dataset(args.dataset_dir)
	perfect_trajectories = load_perfect_trajectories(args.dataset_dir)

	start_time = time.time()
	
	# Initialize training info dictionary
	training_info = {
		'critic_loss': [],
        'actor_loss': [],
        'q_mean': [],
        'q_max': [],
        'q_min': [],
        'mse_from_perfect': [],
        'bc_loss': [],
        'q_value': [],
        'target_q_value': []
		}
	
	# Main training loop
	for t in range(int(args.max_timesteps+1)):
		if t % args.eval_freq == 0:
			evaluate_and_print(RL_agent, t, start_time, args, training_info, perfect_trajectories)
			
			# Save model every 50k iterations
			if t % 50000 == 0:
				RL_agent.save(f"{args.save_dir}/model_iter_{t}")
				print(f"Model saved at iteration {t}")

		info = RL_agent.train()
		
		# Update training info
		for key, value in info.items():
			if value is not None:
				training_info[key].append(value)

	# # Save the final model
	# RL_agent.save(f"{args.save_dir}/final_model")
    
    # Save training info
	with open(f"{args.save_dir}/training_info.pkl", 'wb') as f:
		pickle.dump(training_info, f)
		
	# Plot training curves
	plot_training_curves(training_info, args.save_dir)

def evaluate_on_perfect_trajectories(RL_agent, perfect_trajectories):
	# Evaluate the agent's performance on perfect trajectories
    total_mse = 0
    for trajectory in perfect_trajectories:
        for state, perfect_action in trajectory:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(RL_agent.device)
            with torch.no_grad():
                zs = RL_agent.encoder.zs(state_tensor)
                predicted_action = RL_agent.actor(state_tensor, zs).cpu().numpy().flatten()
            mse = np.mean((predicted_action - perfect_action)**2)
            total_mse += mse
    return total_mse / len(perfect_trajectories)

def load_perfect_trajectories(dataset_dir, num_perfect_trajectories=6):
	# Load perfect trajectories from the dataset
	perfect_trajectories = []
	for i in range(num_perfect_trajectories):
		if i < 10:
			file_path = os.path.join(dataset_dir, f'trajectory_0{i}.h5')
		else:
			file_path = os.path.join(dataset_dir, f'trajectory_{i}.h5')
		with h5py.File(file_path, 'r') as f:
			trajectory = []
			for step in f.values():
				state = step['state'][()]
				action = step['action'][()]
				trajectory.append((state, action))
			perfect_trajectories.append(trajectory)
	return perfect_trajectories

def compute_additional_metrics(RL_agent):
	# Compute additional metrics for evaluation
    states, actions, _, _, _ = RL_agent.replay_buffer.get_all_data()
    
    with torch.no_grad():
        zs = RL_agent.encoder.zs(states)
        predicted_actions = RL_agent.actor(states, zs)
        zsa = RL_agent.encoder.zsa(zs, actions)
        q_values = RL_agent.critic(states, actions, zsa, zs)
    
    bc_loss = F.mse_loss(predicted_actions, actions)
    q_mean = q_values.mean().item()
    q_max = q_values.max().item()
    q_min = q_values.min().item()
    
    action_mean = actions.mean(dim=0)
    action_std = actions.std(dim=0)
    predicted_action_mean = predicted_actions.mean(dim=0)
    predicted_action_std = predicted_actions.std(dim=0)
    
    return {
        'bc_loss': bc_loss.item(),
        'q_mean': q_mean,
        'q_max': q_max,
        'q_min': q_min,
        'action_mean': action_mean.cpu().numpy(),
        'action_std': action_std.cpu().numpy(),
        'predicted_action_mean': predicted_action_mean.cpu().numpy(),
        'predicted_action_std': predicted_action_std.cpu().numpy(),
    }	

def evaluate_and_print(RL_agent, t, start_time, args, training_info, perfect_trajectories):
	print("---------------------------------------")
	print(f"Evaluation at {t} time steps")
	print(f"Total time passed: {round((time.time()-start_time)/60.,2)} min(s)")
	
	mse = evaluate_on_perfect_trajectories(RL_agent, perfect_trajectories)
	print(f"MSE from perfect trajectories: {mse:.5f}")
	
	additional_metrics = compute_additional_metrics(RL_agent)
	print(f"Behavioral Cloning Loss: {additional_metrics['bc_loss']:.5f}")
	print(f"Q-value - Mean: {additional_metrics['q_mean']:.5f}, Max: {additional_metrics['q_max']:.5f}, Min: {additional_metrics['q_min']:.5f}")

	training_info['mse_from_perfect'].append(mse)
	training_info['bc_loss'].append(additional_metrics['bc_loss'])
	training_info['q_mean'].append(additional_metrics['q_mean'])
	training_info['q_max'].append(additional_metrics['q_max'])
	training_info['q_min'].append(additional_metrics['q_min'])
	
	
	if training_info['critic_loss']:
		print(f"Average critic loss: {np.mean(training_info['critic_loss'][-args.eval_freq:]):.3f}")
	if training_info['actor_loss']:
		print(f"Average actor loss: {np.mean([l for l in training_info['actor_loss'][-args.eval_freq:] if l is not None]):.3f}")
	if training_info['q_value']:
		print(f"Average Q-value: {np.mean(training_info['q_value'][-args.eval_freq:]):.3f}")
	if training_info['target_q_value']:
		print(f"Average target Q-value: {np.mean(training_info['target_q_value'][-args.eval_freq:]):.3f}")
	print("---------------------------------------")


def plot_training_curves(training_info, save_dir):
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    
    axs[0, 0].plot(training_info['critic_loss'])
    axs[0, 0].set_title('Critic Loss')
    
    axs[0, 1].plot(training_info['actor_loss'])
    axs[0, 1].set_title('Actor Loss')
    
    axs[1, 0].plot(training_info['q_mean'])
    axs[1, 0].set_title('Mean Q-Value')
    
    axs[1, 1].plot(training_info['mse_from_perfect'])
    axs[1, 1].set_title('MSE from Perfect Trajectories')
    
    axs[2, 0].plot(training_info['bc_loss'])
    axs[2, 0].set_title('Behavioral Cloning Loss')
    
    axs[2, 1].plot(training_info['q_value'], label='Q-Value')
    axs[2, 1].plot(training_info['target_q_value'], label='Target Q-Value')
    axs[2, 1].set_title('Q-Value vs Target Q-Value')
    axs[2, 1].legend()
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/training_curves.png")
    plt.close()

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset_dir", default="Dataset1", type=str)
	parser.add_argument("--seed", default=0, type=int)
	parser.add_argument("--eval_freq", default=5000, type=int)
	parser.add_argument("--max_timesteps", default=50000, type=int)
	parser.add_argument("--save_dir", default="./saved_models", type=str)
	args = parser.parse_args()

	# Set random seed
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)

	# Create save directory if it doesn't exist
	os.makedirs(args.save_dir, exist_ok=True)

	state_dim = 10 
	action_dim = 2  
	max_action = np.array([2.0, 0.8])  

	RL_agent = TD7.Agent(state_dim, action_dim, max_action, offline=True)

	train_offline(RL_agent, args)