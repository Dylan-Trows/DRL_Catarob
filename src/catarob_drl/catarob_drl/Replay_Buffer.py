import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=1e6):
        self.max_size = int(max_size)
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, action_dim))
        self.next_state = np.zeros((self.max_size, state_dim))
        self.reward = np.zeros((self.max_size, 1))                  #np.zeros(self.max_size)
        self.not_done = np.zeros((self.max_size, 1))                #np.zeros(self.max_size, dtype=np.bool)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done                         # use np.bool

        self.ptr = (self.ptr + 1) % self.max_size                   # loop back to the start of the array
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        batch = np.random.randint(0, self.size, size=batch_size)    #np.random.choice(self.size, batch_size)

        # states = self.state[batch]
        # next_states = self.next_state[batch]
        # actions = self.action[batch]
        # rewards = self.reward[batch]
        # not_dones = self.not_done[batch]
        return (
            torch.FloatTensor(self.state[batch]),
            torch.FloatTensor(self.action[batch]),
            torch.FloatTensor(self.next_state[batch]),
            torch.FloatTensor(self.reward[batch]),
            torch.FloatTensor(self.not_done[batch])
        )
