"""
MLP agent that tries to minimize the binary cross entropy error with respect to the actions taken.
"""
import torch
import torch.nn as nn
import argparse
import gym
import d4rl
import torch.optim as optim
import numpy as np
import pickle

from torch.utils.data import DataLoader
from src.datasets.demonstrations_dataset import DemonstrationsDataset
from src.datasets.d4rl_dataset import D4RLDataset


parser = argparse.ArgumentParser(description='Behavioural Cloning')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
parser.add_argument('--max_iters', type=int, default=1000000)
parser.add_argument('--env', type=str, default='cartpole-v0')
parser.add_argument('--demonstrations_path', type=str, default='demos.pickle')
parser.add_argument('--n_epochs', type=int, default=100)
args = parser.parse_args()


class MLP(nn.Module):
    """
    MLP agent that tries to minimize the binary cross entropy error with respect to the actions taken.
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(4, 128)
        self.activation = nn.ReLU()
        self.fc2 = nn.Linear(128, 2)
    
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.fc2(x)
        return x


if args.env == 'maze2d-open-v0':
    env = gym.make('maze2d-open-v0')
    dataset = D4RLDataset(env.get_dataset())
    criterion = torch.nn.functional.huber_loss
else:
    # Open pickle file
    with open(args.demonstrations_path, 'rb') as f:
        data = pickle.load(f)
    dataset = DemonstrationsDataset(data)
    criterion = torch.nn.functional.binary_cross_entropy

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Seed and take length of dataset
len_ds = len(dataset)
# env.seed(args.seed)
torch.manual_seed(args.seed)

# Create model and optimizer
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
eps = np.finfo(np.float32).eps.item()

losses = []

# TODO: check if everything is working
def train():
    for epoch in range(args.n_epochs):
        for i, data in enumerate(dataloader):
            state = data['state']
            action = data['action']
            chosen_action = model(state)
            loss = criterion(chosen_action, action)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch % args.log_interval == 0:
            print('Iteration: {}\tRunning Loss: {}'.format(i, np.mean(losses)))

        # TODO: Add testing by interacting with the environment
    
    torch.save(model.state_dict(), 'bc_model.pt')

if __name__ == '__main__':
    train()


