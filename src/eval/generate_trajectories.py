import numpy as np
import gym
import gym_minigrid
from src.wrappers.wrapper import FrameStack
from collections import defaultdict
import pickle
import torch
import argparse

parser = argparse.ArgumentParser(description='Take trajectories made by a policy.')
parser.add_argument('--seed', type=int, default=42, metavar='N',
                    help='random seed (default: 42)')
parser.add_argument('--render', type=bool, default=False,
                    help='render the environment')
parser.add_argument('--nb_dem', type=int, default=100, metavar='N',
                    help='number of demonstrations (default: 100)')                
parser.add_argument('--save', type=bool, default=False,
                    help='save trajectories')
parser.add_argument('--save_path', type=str, default='generated_trajectories.pickle',
                    help='path to save the demonstrations')
parser.add_argument('--generator', type=str, default='almost-optimal',
                    help='almost-optimal or sub-optimal')
parser.add_argument('--env', type=str, default='MiniGrid-Empty-8x8-v0')
args = parser.parse_args()


def generate_almost_optimal(env, env_frame, seed, nb_demos, device, render=False):
    """
    Generate optimal trajectories (turn around correctly and use the less number of steps possible)
    """
    trajectories = []

    env.seed(seed)
    env_frame.seed(seed)
    count = 0

    # left, right, forward
    actions = [0,1,2]

    for i in range(nb_demos):
        env.reset()
        state = torch.from_numpy(env_frame.reset()).to(device).unsqueeze(0)
        trajectory = defaultdict(list)
        
        if render:
            env.render()
        go_right = env.unwrapped.grid.height - 3
        go_down = go_right
        
        # we can only stay down or right
        dir = 'right'
        done = False

        while not done:
            ch = np.random.choice(actions)

            if dir == 'right' and ch == 2 and go_right > 0:
                go_right -= 1
            elif dir == 'down' and ch == 2 and go_down > 0:
                go_down -= 1
            elif ch == 0 and dir == 'down':
                dir = 'right'
            elif ch == 1 and dir == 'right':
                dir = 'down'
            else:
                continue

            count += 1
            _, _, done, _ = env.step(ch)
            next_state, reward, _, _ = env_frame.step(ch)
            next_state = torch.from_numpy(next_state).to(device).unsqueeze(0)

            trajectory['state'].append(state)
            trajectory['action'].append(torch.Tensor([ch]).long().to(device).unsqueeze(-1))
            trajectory['next_state'].append(next_state)
            trajectory['reward'].append(torch.Tensor([reward]).float().to(device))

            state = next_state.clone()
            if render:
                env.render()

        
        print('nb of transitions so far: ', count)

        trajectories.append(trajectory)

    return trajectories

def generate_sub_optimal(env, env_frame, seed, nb_demos, device, render=False):
    """
    Generate sub-optimal trajectories. (turn around and then always reduce the distance to the goal)
    """
    trajectories = []
    env.seed(seed)
    env_frame.seed(seed)

    if render:
        env.render()

    count = 0

    for i in range(nb_demos):
        trajectory = defaultdict(list)
        state = env.reset()
        state = torch.from_numpy(env_frame.reset()).to(device).unsqueeze(0)

        if render:
            env.render()
        past_actions = np.array([1, 1, 1])
        sample_probas = np.array([0.10, 0.10, 0.5])

        done = False
        while not done:
            count += 1
            action_p = (sample_probas / past_actions) / sum(sample_probas / past_actions)
            rn = np.random.rand()
            i = 0
            cum_sum = 0

            while cum_sum < rn:
                cum_sum += action_p[i]
                i += 1

            action = i - 1

            next_state, reward, done, _ = env.step(action)
            next_state, reward, _, _ = env_frame.step(action)
            next_state = torch.from_numpy(next_state).to(device).unsqueeze(0)
            
            trajectory['state'].append(state)
            trajectory['action'].append(torch.Tensor([action]).long().to(device).unsqueeze(-1))
            trajectory['next_state'].append(next_state)
            trajectory['reward'].append(torch.Tensor([reward]).float().to(device))

            state = next_state.clone()
            if render:
                env.render()

        trajectories.append(trajectory)
        print('nb of transitions so far: ', count)

    return trajectories


if __name__ == '__main__':
    np.random.seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = gym.make(args.env)
    env_frame = FrameStack(gym.make(args.env))

    if args.generator == "almost-optimal":
        trajectories = generate_almost_optimal(env, env_frame, args.seed, args.nb_dem, device, args.render)

    elif args.generator == "sub-optimal":
        trajectories = generate_sub_optimal(env, env_frame, args.seed, args.nb_dem, device, args.render)
    
    else:
        raise ValueError("Generator not recognized")

    # And then we save the trajectories specified
    if args.save:
        with open(args.save_path, 'wb') as f:
            pickle.dump(trajectories, f)

        print('Saving as ' + args.save_path)
