"""
Collect trajectories from a policy.
"""

import argparse
import gym
import numpy as np
from collections import defaultdict
import d4rl

import torch
from agent.ac.model import select_action, Policy
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
parser.add_argument('--save_file', type=str, default='demos',
                    help='path to save the demonstrations')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

def main():
    model = Policy()
    model.load_state_dict(torch.load(args.path_agent))
    # model.eval()
    transitions = defaultdict(list)

    # run inifinitely many episodes
    for i_episode in range(args.nb_dem):

        # reset environment and episode reward
        state = env.reset()
        
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        for t in range(1, 10000):
            last_state = state.copy()

            # select action from policy
            action = select_action(state, model)

            # take the action
            state, reward, done, _ = env.step(action)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward

            # append transitions to transitions list
            transitions['state'].append(last_state)
            transitions['new_state'].append(state)
            transitions['reward'].append(reward)
            transitions['action'].append(action)


            if done:
                break

        # log results
        nb_transitions = len(model.rewards)

        print('Episode {}\tNb of transitions {}\tLast reward: {:.2f}'.format(
                i_episode, nb_transitions, ep_reward))

    # Now we save this data
    with open(args.save_file + '.pickle', 'wb') as f:
        pickle.dump(transitions, f)


if __name__ == '__main__':
    main()