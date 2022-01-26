"""
Random Agent.
"""

import torch
import torch.nn as nn
from torch.nn.functional import one_hot


class RandomAgent(nn.Module):
    def __init__(self, action_space='discrete', n_actions=2, action_range=(-1, 1)):
        super(RandomAgent, self).__init__()
        self.action_space = action_space
        self.n_actions = n_actions
        self.action_range = action_range

        # action & reward buffer for evaluating policy's performance
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        if self.action_space == 'discrete':
            return one_hot(torch.randint(0, self.n_actions, (x.shape[0],)), num_classes=self.n_actions).float()

        else:
            return torch.rand(x.shape[0], self.n_actions) * (self.action_range[1] - self.action_range[0]) + self.action_range[0]
        