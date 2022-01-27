"""
Double Deep-Q Learning agent with prioritized experience replay.

Based on pytorch tutorial on dqns.
We will use the images as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import math
from gym_minigrid.minigrid import OBJECT_TO_IDX


class DQN(nn.Module):

    def __init__(self, h, w, n_actions, device, eps_start=0.4, eps_end=0.01, frac_eps=0.5, kernel_size=3, stride=1):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=kernel_size, stride=stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride)
        self.bn2 = nn.BatchNorm2d(32)
        # self.conv3 = nn.Conv2d(32, 32, kernel_size=kernel_size, stride=stride)
        # self.bn3 = nn.BatchNorm2d(32)
        self.embed = nn.Embedding(len(OBJECT_TO_IDX), 3)
        self.device = device
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.frac_eps = frac_eps
        self.n_actions = n_actions

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = kernel_size, stride = stride):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(w))
        convh = conv2d_size_out(conv2d_size_out(h))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.long().to(self.device)
        x = self.embed(x).flatten(-2).permute(0,3,1,2)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        # x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

    def select_action(self, curr_step, state, nb_transitions):
        sample = random.random()
        # TODO: try with linear decay and control randomness with a fraction of the training
        # that should be for focusing on exploration.
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * max(0, 1 - curr_step / (nb_transitions * self.frac_eps))
        # eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-1. * curr_step / self.eps_decay)
        # print(eps_threshold)
        if sample > eps_threshold:
            with torch.no_grad():
                return self.forward(state).max(1)[1].view(1, 1)
        
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long, device=self.device)