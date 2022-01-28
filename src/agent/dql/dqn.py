"""
Double Deep-Q Learning agent with prioritized experience replay.

Based on pytorch tutorial on dqns.
We will use the images as input.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from gym_minigrid.minigrid import OBJECT_TO_IDX
from torch.distributions import Categorical
import numpy as np

class DQN(nn.Module):

    # frac_eps 0.4
    def __init__(self, h, w, n_actions, device, eps_start=1.0, eps_end=0.01, frac_eps=0.4, 
                kernel_size=2, stride=2, padding=1, embed_size=3):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(12, 16, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3 = nn.BatchNorm2d(64)
        # We add 1 so that we consider the padding
        self.embed = nn.Embedding(len(OBJECT_TO_IDX) + 1, embed_size)
        self.device = device
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.frac_eps = frac_eps
        self.n_actions = n_actions

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = kernel_size, stride = stride, padding=padding):
            return (size + 2 * padding - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 64
        self.head = nn.Linear(linear_input_size, 64)
        self.head2 = nn.Linear(64, n_actions)
        self.rewards = []
        self.saved_actions = []

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = x.long().to(self.device)
        x = self.embed(x).flatten(-2).permute(0,3,1,2)
        x = F.elu(self.bn1(self.conv1(x)))
        x = F.elu(self.bn2(self.conv2(x)))
        x = F.elu(self.bn3(self.conv3(x)))
        return self.head2(F.relu(self.head(x.view(x.size(0), -1))))

    def select_action(self, curr_step, state, nb_transitions):
        sample = random.random()
        # Linear Decay
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * max(0, 1 - curr_step / (nb_transitions * self.frac_eps))
        
        if sample > eps_threshold:
            with torch.no_grad():
                return self.forward(state).max(1)[1].view(1, 1)
        
        else:
            return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long, device=self.device)

    def select_greedy(self, state):
        with torch.no_grad():
            q_values = self.forward(state)
            return torch.max(q_values, 1, keepdim=True)[1]

    def collect_demos(self, state, random_chance=0.3):
        with torch.no_grad():
            rand_nb = np.random.rand()

            if rand_nb > random_chance:
                return self.select_greedy(state)
            else:
                return torch.tensor([[random.randrange(self.n_actions)]], dtype=torch.long, device=self.device)
            # q_values = self.forward(state)
            # probs = F.softmax(q_values / temp, dim=1)

            # # create a categorical distribution over the list of probabilities of actions
            # m = Categorical(probs)

            # # and sample an action using the distribution
            # action = m.sample()
            # return action