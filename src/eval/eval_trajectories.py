"""
Collect trajectories from a policy.
"""

import argparse
import gym
import numpy as np
from collections import defaultdict
import d4rl

import torch
from src.agent.ac.ac import select_greedy_action
from src.agent.agents import mapping_models
import pickle


parser = argparse.ArgumentParser(description='Take trajectories made by a policy.')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--nb_dem', type=int, default=10, metavar='N',
                    help='number of demonstrations (default: 10)')                
parser.add_argument('--noise', type=float, default=None, metavar='G',
                    help='noise to be added on state seen by agent')
parser.add_argument('--path_agent', type=str, default='a2c.pt',
                    help='path to agent to be evaluated')
parser.add_argument('--save', type=bool, default=False,
                    help='save trajectories')
parser.add_argument('--save_file', type=str, default='demos',
                    help='path to save the demonstrations')
parser.add_argument('--model', type=str, default='ac',
                    help='model being evaluated')
args = parser.parse_args()


CurrentModel = mapping_models[args.model]


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

def main():
    model = CurrentModel()
    model.load_state_dict(torch.load(args.path_agent))
    # TODO: set model.eval() when necessary
    transitions = defaultdict(list)
    trajectories = []

    # run inifinitely many episodes
    for i_episode in range(args.nb_dem):

        traj_transitions = defaultdict(list)

        # reset environment and episode reward
        state = env.reset()
        
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        for t in range(1, 10000):
            last_state = state.copy()

            # select action from policy
            action = select_greedy_action(state, model)

            # take the action
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward

            # append transitions to transitions list
            traj_transitions['state'].append(last_state)
            traj_transitions['new_state'].append(state)
            traj_transitions['reward'].append(reward)
            traj_transitions['action'].append(action)

            trajectories.append(traj_transitions.copy())

            if done:
                transitions = transitions | traj_transitions
                break

        # log results
        nb_transitions = len(model.rewards)

        print('Episode {}\tNb of transitions {}\tLast reward: {:.2f}'.format(
                i_episode, nb_transitions, ep_reward))

    if args.save:
        print("Saving demonstrations...")
        # Now we save the whole data without specific trajectories
        with open(args.save_file + '.pickle', 'wb') as f:
            pickle.dump(transitions, f)

        # And then we save the trajectories specified
        with open(args.save_file + '_trajectories.pickle', 'wb') as f:
            pickle.dump(trajectories, f)

if __name__ == '__main__':
    main()