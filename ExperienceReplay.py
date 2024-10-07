import collections
import torch
import numpy as np

Transition = collections.namedtuple('Experience',
                                    field_names=['current_state', 'action',
                                                 'next_state', 'reward',
                                                 'is_game_on'])

class ExperienceReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = collections.deque(maxlen=capacity)
    def __len__(self):
        return len(self.memory)
    def push(self, transition : Transition):
        self.memory.append(transition)
    def get_sample(self, batch_size, device="cpu"):
        random_indices = np.random.choice(len(self.memory), batch_size, replace = False)
        current_states, actions, next_states, rewards, isgameons = zip(*[self.memory[idx] for idx in random_indices])
        return torch.Tensor(current_states).type(torch.float).to(device), \
                torch.Tensor(actions).type(torch.long).to(device), \
                torch.Tensor(next_states).to(device), \
                torch.Tensor(rewards).to(device), \
                torch.Tensor(isgameons).to(device)