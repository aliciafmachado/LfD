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


parser = argparse.ArgumentParser(description='Behavioural Cloning')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10000, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
parser.add_argument('--max_iters', type=int, default=1000000)
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


env = gym.make('maze2d-open-v0')
env.seed(args.seed)
dataset = env.get_dataset()
len_ds = len(dataset['observations'])
torch.manual_seed(args.seed)
model = MLP()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
eps = np.finfo(np.float32).eps.item()
criterion = torch.nn.functional.huber_loss
losses = []

def train():
    for i in range(args.max_iters):
        j = i % len_ds
        obs = dataset['observations'][j]
        action = dataset['actions'][j]
        chosen_action = model(torch.from_numpy(obs).float())
        loss = criterion(chosen_action, torch.from_numpy(action).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i % args.log_interval == 0:
            print('Iteration: {}\tRunning Loss: {}'.format(i, np.mean(losses)))
    
    torch.save(model.state_dict(), 'model.pt')

if __name__ == '__main__':
    train()


