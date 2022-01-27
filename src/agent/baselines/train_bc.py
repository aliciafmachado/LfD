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
from src.agent.agents import mapping_models
from src.agent.actor_critic.ac import select_greedy_action


parser = argparse.ArgumentParser(description='Behavioural Cloning')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
parser.add_argument('--max_iters', type=int, default=1000000)
parser.add_argument('--env', type=str, default='CartPole-v0')
parser.add_argument('--demonstrations_path', type=str, default='demos.pickle')
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--save_path', type=str, default='model_bc.pt')
parser.add_argument('--model', type=str, default='bc')
args = parser.parse_args()


if args.env == 'maze2d-open-v0':
    env = gym.make('maze2d-open-v0')
    dataset = D4RLDataset(env.get_dataset())
    criterion = torch.nn.functional.huber_loss
else:
    env = gym.make(args.env)
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
model = mapping_models[args.model]()
if len(list(model.parameters())) != 0:
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

eps = np.finfo(np.float32).eps.item()

losses = []

# TODO: check if everything is working
def train():
    for epoch in range(args.n_epochs):
        accuracy = []
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
            accuracy.append(np.mean(np.argmax(chosen_action_probs.detach().numpy(), axis=1) == action.numpy()))

        if epoch % args.log_interval == 0:
            print('Iteration: {}\tRunning Loss: {}'.format(epoch, np.mean(losses)))
            print('Iteration: {}\tMean accuracy: {}'.format(epoch, np.mean(accuracy)))
            # reset environment and episode reward
            # state = env.reset()
            
            # ep_reward = 0

            # # for each episode, only run 9999 steps so that we don't 
            # # infinite loop while learning
            # for t in range(1, 10000):
            #     # select action from policy
            #     action = select_greedy_action(state, model)

            #     # take the action
            #     state, reward, done, _ = env.step(action)

            #     if args.render:
            #         env.render()

            #     model.rewards.append(reward)
            #     ep_reward += reward

            #     if done:
            #         break

        # TODO: Add testing by interacting with the environment
    
    print("Saving model as " + args.save_path)
    torch.save(model.state_dict(), args.save_path)

if __name__ == '__main__':
    train()


