import torch 
import torch.nn as nn
import torchvision.transforms as T
import argparse
import gym
import torch.optim as optim
import gym_minigrid
import numpy as np

from src.utils.replay import ReplayMemory
from src.utils.transition import Transition
from src.agent.dql.dqn import DQN
from gym_minigrid.wrappers import StateBonus
from matplotlib import pyplot as plt

# from collections import namedtuple, deque
from PIL import Image

parser = argparse.ArgumentParser(description='DQN training')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--lr', type=float, default=0.0005, metavar='LR')
parser.add_argument('--nb_episodes', type=int, default=100)
parser.add_argument('--nb_transitions', type=int, default=25000)
parser.add_argument('--env', type=str, default='MiniGrid-Empty-5x5-v0')
parser.add_argument('--save_path', type=str, default='dqn.pt')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--target_update', type=int, default=1000)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--render', type=bool, default=False)
parser.add_argument('--state_bonus', type=bool, default=False)
args = parser.parse_args()


def optimize(policy_net, target_net, optimizer, memory):
    if len(memory) < args.batch_size:
        return

    transitions = memory.sample(args.batch_size)

    # Transpose the batch
    batch = Transition(*zip(*transitions))

    # TODO: recheck here
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
    
    # next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    
    # Double Q learning
    # next_state_values[non_final_mask] = next_state_values[non_final_mask].gather(1, target_net(non_final_next_states).max(1)[1])

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

    return loss.detach().cpu().item()


def preprocess(state, device):
    return torch.from_numpy(state['image'][:,:,0:1]).to(device).unsqueeze(0)


def evaluate(model, env, nb_episodes=10):
    """
    Evaluate a trained model on the environment
    """
    rewards = []
    for i in range(nb_episodes):
        state = preprocess(env.reset(), model.device)
        done = False
        total_reward = 0
        t = 0
        while not done:
            action = model.select_greedy(state)
            temp_state, reward, done, _ = env.step(action)
            state = preprocess(temp_state, model.device)
            total_reward += reward
            t += 1
        
        rewards.append(total_reward)
    return np.mean(rewards)


def train():
    # Create environment
    env = gym.make(args.env)
    eval_env = gym.make(args.env)

    if args.state_bonus:
        print('Using state bonus.')
        env = StateBonus(env)
    
    env.seed(args.seed)
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
    optimizer = optim.RMSprop(policy_net.parameters())

    # Create replay memory
    memory = ReplayMemory(50000)

    rewards_per_ep = []
    ep_reward = 0.
    all_rewards = []
    losses = []

    # We just take the index of objects for now
    state = preprocess(env.reset(), device)
    reward_eval = []

    for step in range(args.nb_transitions):

        action = policy_net.select_action(step, state, nb_transitions=args.nb_transitions)
        temp_state, reward_, done, _ = env.step(action.item())
        next_state = preprocess(temp_state, device)

        reward = torch.tensor([reward_], device=device)

        if not done:
            memory.push(state, action, next_state, reward)
        else:
            memory.push(state, action, None, reward)
            
        # Move to next state
        state = next_state

        # Update the reward
        ep_reward += reward_
        all_rewards.append(reward_)

        # Call the optimization function to do backprop
        loss = optimize(policy_net, target_net, optimizer, memory)
        
        if loss is not None:
            losses.append(loss)

        if args.render:
            env.render()

        if done: 
            state = torch.from_numpy(env.reset()['image'][:,:,0:1]).to(device).unsqueeze(0)
            rewards_per_ep.append(ep_reward)
            ep_reward = 0

        if (step + 1) % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (step + 1) % args.log_interval == 0:
            mean_loss = 0
            if len(losses) > 0:
                mean_loss = np.mean(losses)

            reward_eval.append(evaluate(policy_net, eval_env, nb_episodes=5))

            print('Step {}\tEval reward: {:.2f}\tAverage reward in last 10 eps: {:.2f}\tLoss: {:.3f}'.format(
                step+1, reward_eval[-1], np.mean(rewards_per_ep[-10:]), mean_loss))


    print("Finished training")
    print("Saving model to {}".format(args.save_path))
    torch.save(policy_net.state_dict(), args.save_path)

    # Saving rewards on eval
    plt.plot(reward_eval)
    plt.savefig(args.save_path + '_rewards.png')


if __name__ == '__main__':
    train()

