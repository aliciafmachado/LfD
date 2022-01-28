"""
Classic Replay Buffer.
"""

import random
from src.utils.transition import Transition
from collections import deque

class ReplayMemory(object):
    def __init__(self, capacity=100000, initial_data={}, dem_trajectory=False, dem_factor=0.1, dem_factor_decay=0.):
        super(ReplayMemory, self).__init__()
        self.memory = deque([], maxlen=capacity)
        
        # TODO: change demonstrations saved to a list of transitions
        self.dem_memory = deque([Transition(*tr) for tr in zip(*initial_data.values())], maxlen=capacity)
        self.use_demonstrations = len(initial_data['reward']) > 0
        self.dem_factor = dem_factor

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample_demonstrations(self, batch_size):
        return random.sample(self.dem_memory, batch_size)

    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample(self, batch_size):
        batch_dem = int(self.dem_factor * batch_size)

        # If we don't have enough data available to follow the proportions
        # we use only what we have and complete with demonstrations
        if batch_size - batch_dem > len(self.memory): 
            batch_samp = len(self.memory)
            batch_dem = batch_size - batch_samp
        
        else:
            batch_samp = batch_size - batch_dem

        sample_dem = self.sample_demonstrations(batch_dem)
        sample_mem = self.sample_memory(batch_samp)
        return sample_dem + sample_mem

    def __len__(self):
        return len(self.memory) + len(self.dem_memory)