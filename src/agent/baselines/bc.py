import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """
    MLP agent that tries to minimize the binary cross entropy error with respect to the actions taken.
    """
    def __init__(self, action_space='discrete'):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
        self.softmax = F.softmax
        self.action_space = action_space

        # action & reward buffer for evaluating policy's performance
        self.saved_actions = []
        self.rewards = []
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        if self.action_space == 'discrete':
            x = self.softmax(x)
        return x