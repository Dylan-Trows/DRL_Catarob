import numpy as np
import torch
import h5py
import os

class LAP(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		device,
		max_size=1e6,
		batch_size=256,
		max_action=1,
		normalize_actions=True,
		prioritized=True
	):
		# Initialize buffer parameters
		max_size = int(max_size)
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.device = device
		self.batch_size = batch_size

		# Initialize numpy arrays to store transitions
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		# Initialize prioritized experience replay if enabled
		self.prioritized = prioritized
		if prioritized:
			self.priority = torch.zeros(max_size, device=device)
			self.max_priority = 1

		# Store max_action for action normalization
		self.normalize_actions = max_action if normalize_actions else 1

	
	def add(self, state, action, next_state, reward, done):
		# Add a new transition to the buffer
		self.state[self.ptr] = state
		self.action[self.ptr] = action/self.normalize_actions
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		
		if self.prioritized:
			self.priority[self.ptr] = self.max_priority

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self):
		# Sample a batch of transitions
		if self.prioritized:
			csum = torch.cumsum(self.priority[:self.size], 0)
			val = torch.rand(size=(self.batch_size,), device=self.device)*csum[-1]
			self.ind = torch.searchsorted(csum, val).cpu().data.numpy()
		else:
			self.ind = np.random.randint(0, self.size, size=self.batch_size)

		return (
			torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.not_done[self.ind], dtype=torch.float, device=self.device)
		)


	def update_priority(self, priority):
		# Update priorities for prioritized experience replay
		self.priority[self.ind] = priority.reshape(-1).detach()
		self.max_priority = max(float(priority.max()), self.max_priority)


	def reset_max_priority(self):
		# Reset the max priority
		self.max_priority = float(self.priority[:self.size].max())


	def load_custom_dataset(self, dataset_dir):
		# Load custom dataset from h5 files
		states, actions, rewards, next_states, dones = [], [], [], [], []
		
		for filename in os.listdir(dataset_dir):
			if filename.endswith('.h5'):
				filepath = os.path.join(dataset_dir, filename)
				with h5py.File(filepath, 'r') as hf:
					for step_key in hf.keys():
						step = hf[step_key]
						states.append(step['state'][()])
						actions.append(step['action'][()])
						rewards.append(step['reward'][()])
						next_states.append(step['next_state'][()])
						dones.append(step['done'][()])
		
		self.state = np.array(states)
		self.action = np.array(actions) / self.normalize_actions
		self.next_state = np.array(next_states)
		self.reward = np.array(rewards).reshape(-1, 1)
		self.not_done = 1. - np.array(dones).reshape(-1, 1)
		self.size = self.state.shape[0]
		
		if self.prioritized:
			self.priority = torch.ones(self.size).to(self.device)

	def get_all_data(self):
		# Return all data in the buffer
		return (
			torch.tensor(self.state, dtype=torch.float, device=self.device),
			torch.tensor(self.action, dtype=torch.float, device=self.device),
			torch.tensor(self.next_state, dtype=torch.float, device=self.device),
			torch.tensor(self.reward, dtype=torch.float, device=self.device),
			torch.tensor(self.not_done, dtype=torch.float, device=self.device)
		)