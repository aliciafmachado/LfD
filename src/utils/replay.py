"""
Usual Replay Buffer.
"""

import random
from src.utils.transition import Transition
from collections import deque

class ReplayMemory(object):
    def __init__(self, capacity=100000):
        super(ReplayMemory, self).__init__()
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)