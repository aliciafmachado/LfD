"""
Collect trajectories from a policy.
"""

import argparse
import gym
import numpy as np
from collections import defaultdict
import d4rl
import gym_minigrid

import torch
from src.agent.utils import select_greedy_action
from src.agent.agents import mapping_models
import pickle
from src.wrappers.wrapper import FrameStack


parser = argparse.ArgumentParser(description='Take trajectories made by a policy.')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--nb_dem', type=int, default=10, metavar='N',
                    help='number of demonstrations (default: 10)')                
parser.add_argument('--noise', type=float, default=0.,
                    help='noise to be added on state seen by agent')
parser.add_argument('--path_agent', type=str, default='a2c.pt',
                    help='path to agent to be evaluated')
parser.add_argument('--save', type=bool, default=False,
                    help='save trajectories')
parser.add_argument('--save_file', type=str, default='demos',
                    help='path to save the demonstrations')
parser.add_argument('--model', type=str, default='ac',
                    help='model being evaluated')
parser.add_argument('--env', type=str, default='MiniGrid-Empty-5x5-v0')
args = parser.parse_args()


CurrentModel = mapping_models[args.model]

if args.env == 'MiniGrid-Empty-5x5-v0':
    env = FrameStack(gym.make('MiniGrid-Empty-5x5-v0'))

else:
    env = gym.make(args.env)
    
env.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

def main():

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.model == 'dqn':
        h, w, _ = env.observation_space['image'].shape
        n_actions = env.action_space.n
        model = CurrentModel(h, w, n_actions, device)

    else:
        model = CurrentModel()
    model.load_state_dict(torch.load(args.path_agent, map_location=device))

    transitions = defaultdict(list)
    trajectories = []

    # run inifinitely many episodes
    for i_episode in range(args.nb_dem):

        traj_transitions = defaultdict(list)

        # reset environment and episode reward
        state = torch.from_numpy(env.reset()).to(device).unsqueeze(0)
        
        ep_reward = 0

        # for each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        for t in range(1, 10000):
            last_state = state.clone()

            # select action from policy
            if args.model == 'dqn':
                action = model.collect_demos(state, random_chance=args.noise)
            
            else:
                action = select_greedy_action(state, model)

            # take the action
            temp_state, reward, done, _ = env.step(action)
            state = torch.from_numpy(temp_state).to(device).unsqueeze(0)

            if args.render:
                env.render()

            model.rewards.append(reward)
            ep_reward += reward

            # append transitions to transitions list
            traj_transitions['state'].append(last_state)
            traj_transitions['action'].append(action)
            traj_transitions['new_state'].append(state)
            traj_transitions['reward'].append(torch.Tensor([reward]).to(device))

            trajectories.append(traj_transitions.copy())

            if done:
                transitions['state'].extend(traj_transitions['state'])
                transitions['action'].extend(traj_transitions['action'])
                transitions['new_state'].extend(traj_transitions['new_state'])
                transitions['reward'].extend(traj_transitions['reward'])
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