"""
MLP agent that tries to minimize the binary cross entropy error with respect to the actions taken.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import gym
import d4rl
import torch.optim as optim
import numpy as np
import pickle

from torch.utils.data import DataLoader
from src.datasets.demonstrations_dataset import DemonstrationsDataset
from src.datasets.d4rl_dataset import D4RLDataset
from src.agent.baselines.bc import MLP
from src.agent.ac.ac import select_greedy_action


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
parser.add_argument('--save_path', type=str, default='model_bc.pt')
args = parser.parse_args()


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
        for _, data in enumerate(dataloader):
            state = data['state']
            action = data['action']
            chosen_action_probs = model(state) # torch.argmax(model(state), dim=1)
            # print(select_greedy_action(state.numpy(), model))
            loss = criterion(chosen_action_probs, F.one_hot(action, num_classes=2).squeeze().float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if epoch % args.log_interval == 0:
            print('Iteration: {}\tRunning Loss: {}'.format(epoch, np.mean(losses)))

        # TODO: Add testing by interacting with the environment
    
    print("Saving model as " + args.save_path)
    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    train()


