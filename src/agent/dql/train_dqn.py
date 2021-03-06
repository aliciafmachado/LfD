import torch 
import torch.nn as nn
import torchvision.transforms as T
import argparse
import gym
import torch.optim as optim
import gym_minigrid
import numpy as np
import pickle

from src.utils.replay import ReplayMemory
from src.utils.transition import TransitionTD
from src.agent.dql.dqn import DQN
from gym_minigrid.wrappers import StateBonus
from matplotlib import pyplot as plt
from src.wrappers.wrapper import FrameStack
from collections import defaultdict


parser = argparse.ArgumentParser(description='DQN training')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--nb_episodes', type=int, default=100)
parser.add_argument('--nb_transitions', type=int, default=10000)
parser.add_argument('--env', type=str, default='MiniGrid-Empty-5x5-v0')
parser.add_argument('--save_path', type=str, default='dqn')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--target_update', type=int, default=1000)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--render', type=bool, default=False)
parser.add_argument('--state_bonus', type=bool, default=False)
parser.add_argument('--memory_size', type=int, default=10000)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--weight_decay', type=float, default=0.) # 1e-4
parser.add_argument('--render_eval', type=bool, default=False)
args = parser.parse_args()


def optimize(policy_net, target_net, optimizer, memory, scheduler):
    if len(memory) < args.batch_size:
        return

    transitions = memory.sample(args.batch_size)

    # Transpose the batch
    batch = TransitionTD(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)),
                                          device=policy_net.device, 
                                          dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Now, we compute the state action values
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Now we compute the value function for all states
    next_state_values = torch.zeros(args.batch_size, device=policy_net.device)
    
    next_state_values_actions_p = policy_net(non_final_next_states).max(1)[1]
    next_state_values[non_final_mask] = torch.gather(target_net(non_final_next_states), 1, next_state_values_actions_p.unsqueeze(1)).squeeze().detach()
    
    # Compute the expected Q values using the Bellman equation
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()
    scheduler.step()

    return loss.detach().cpu().item()


def evaluate(model, env, nb_episodes=10):
    """
    Evaluate a trained model on the environment.
    """
    rewards = []
    for i in range(nb_episodes):
        state = torch.from_numpy(env.reset()).to(model.device).unsqueeze(0)
        done = False
        total_reward = 0
        t = 0

        while not done:
            action = model.select_greedy(state)
            temp_state, reward, done, _ = env.step(action)
            state = torch.from_numpy(temp_state).to(model.device).unsqueeze(0)
            total_reward += reward
            t += 1

            if args.render_eval:
                env.render()
        
        rewards.append(total_reward)
    return np.mean(rewards)


def train():
    # Create environment
    env = FrameStack(gym.make(args.env))
    eval_env = FrameStack(gym.make(args.env))

    if args.state_bonus:
        print('Using state bonus.')
        env = StateBonus(env)
    
    env.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    h, w, _ = env.observation_space['image'].shape

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Take number of actions in environment
    n_actions = env.action_space.n

    # Create policy and target nets
    policy_net = DQN(h, w, n_actions, device).to(device)
    target_net = DQN(h, w, n_actions, device).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    # Optimizer
    optimizer = optim.RMSprop(policy_net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.nb_transitions // 3, gamma = 0.1)

    # Create replay memory
    memory = ReplayMemory(args.memory_size)

    rewards_per_ep = []
    ep_reward = 0.
    losses = []

    # We just take the index of objects for now
    state = torch.from_numpy(env.reset()).to(policy_net.device).unsqueeze(0)
    reward_eval = []
    trajectory = defaultdict(list)

    for step in range(args.nb_transitions):

        action = policy_net.select_action(step, state, nb_transitions=args.nb_transitions)
        temp_state, reward_, done, _ = env.step(action.item())
        next_state = torch.from_numpy(temp_state).to(policy_net.device).unsqueeze(0)
        reward = torch.tensor([reward_], device=device)

        if not done:
            trajectory['state'].append(state)
            trajectory['action'].append(action)
            trajectory['next_state'].append(next_state)
            trajectory['reward'].append(reward)

        if done:
            # Add last state and pushes trajectory into memory
            trajectory['state'].append(state)
            trajectory['action'].append(action)
            trajectory['next_state'].append(None)
            trajectory['reward'].append(reward)
            memory.push_trajectory(trajectory.copy())
            trajectory = defaultdict(list)
            
        # Move to next state
        state = next_state

        # Update the reward
        ep_reward += reward_

        # Call the optimization function to do backprop
        loss = optimize(policy_net, target_net, optimizer, memory, scheduler)
        
        if loss is not None:
            losses.append(loss)

        if args.render:
            env.render()

        if done: 
            state = torch.from_numpy(env.reset()).to(device).unsqueeze(0)
            rewards_per_ep.append(ep_reward)
            ep_reward = 0

        if (step + 1) % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (step + 1) % args.log_interval == 0:
            mean_loss = 0
            if len(losses) > 0:
                mean_loss = np.mean(losses)

            reward_eval.append(evaluate(policy_net, eval_env, nb_episodes=1))

            print('Step {}\tEval reward: {:.2f}\tAverage reward in last 10 eps: {:.2f}\tLoss: {:.3f}'.format(
                step+1, reward_eval[-1], np.mean(rewards_per_ep[-10:]), mean_loss))


    print("Finished training")
    print("Saving model to {}".format(args.save_path))
    torch.save(policy_net.state_dict(), args.save_path + '.pt')

    if args.render or args.render_eval:
        plt.close()

    # Saving rewards on eval
    plt.plot(reward_eval)
    plt.savefig(args.save_path + '_rewards.png')

    # Saving losses
    with open(args.save_path + '_rewards.pickle', 'wb') as f:
        pickle.dump(reward_eval, f)

if __name__ == '__main__':
    train()
