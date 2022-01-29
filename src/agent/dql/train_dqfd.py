from pkg_resources import parse_requirements
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
from src.agent.utils import indicator_fn
from collections import defaultdict


parser = argparse.ArgumentParser(description='DQN training')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--nb_episodes', type=int, default=100)
parser.add_argument('--pretrain_steps', type=int, default=1000)
parser.add_argument('--nb_transitions', type=int, default=10000)
parser.add_argument('--env', type=str, default='MiniGrid-Empty-5x5-v0')
parser.add_argument('--save_path', type=str, default='dqn')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--target_update', type=int, default=1000)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--render', type=bool, default=False)
parser.add_argument('--state_bonus', type=bool, default=False)
parser.add_argument('--memory_size', type=int, default=10000)
parser.add_argument('--lr', type=float, default=0.00001)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--lambda1', type=int, default=1)
parser.add_argument('--lambda2', type=int, default=1)
parser.add_argument('--render_eval', type=bool, default=False)
parser.add_argument('--demonstrations_path', type=str, default='demos_dqn.pickle')
parser.add_argument('--use_expert_loss', type=bool, default=False)
parser.add_argument('--use_td_loss', type=bool, default=False)
parser.add_argument('--n_td', type=int, default=4)
parser.add_argument('--reward_extra', type=bool, default=False)
parser.add_argument('--decay_proportional_dem', type=float, default=0.) # 0.07
parser.add_argument('--dem_prop_min', type=float, default=0.1) # 0.05
parser.add_argument('--dem_prop_init', type=float, default=0.1) # 0.15
args = parser.parse_args()


def optimize(policy_net, target_net, optimizer, memory, scheduler, n_actions, only_dem=False):
    if len(memory) < args.batch_size:
        return

    if only_dem:
        transitions = memory.sample_demonstrations(args.batch_size)
    else:
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
    R_cum_batch = torch.cat(batch.R_cum)
    n_next_state_batch = torch.cat(batch.n_next_state)
    gamma_n_batch = torch.Tensor([g for g in torch.from_numpy(np.array(batch.gamma_n))]).to(target_net.device)

    # Now, we compute the state action values
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Now we compute the value function for all states
    next_state_values = torch.zeros(args.batch_size, device=policy_net.device)
    
    next_state_values_actions_p = policy_net(non_final_next_states).max(1)[1]
    next_state_values[non_final_mask] = torch.gather(target_net(non_final_next_states), 
                                                     1, next_state_values_actions_p.unsqueeze(1)).squeeze().detach()

    # Compute the expected Q values using the Bellman equation
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute MSE loss
    criterion = nn.MSELoss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Expert loss didn't help for the minigrid environment so we remove it
    if only_dem and args.use_expert_loss:
        q_loss_expert = state_action_values + indicator_fn(action_batch, n_actions)
        loss_expert = torch.mean(q_loss_expert.max(1)[0] - state_action_values[action_batch])
        loss += loss_expert

    if args.use_td_loss:
        # Compute TD loss
        n_next_state_values_actions_p = policy_net(n_next_state_batch).max(1)[1]
        n_values = torch.gather(target_net(n_next_state_batch), 1, n_next_state_values_actions_p.unsqueeze(1)).squeeze().detach()
        td_loss = torch.abs(state_action_values.squeeze(1) - R_cum_batch - (args.gamma ** gamma_n_batch) * n_values)
        loss += torch.mean(td_loss)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()

    if scheduler is not None:
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


def pretrain(policy_net, target_net, replay, optimizer, n_steps, n_actions, scheduler=None):
    """
    Pretrain on demonstrations without interacting with environment.
    """
    losses_pretrain = []

    for step in range(n_steps):
        loss = optimize(policy_net, target_net, optimizer, replay, scheduler, n_actions=n_actions, only_dem=True)
        losses_pretrain.append(loss)

        if (step + 1) % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if (step + 1) % args.log_interval == 0:
            print('Pretrain: [{}/{}]\tLoss: {:.3f}'.format(
                step + 1, n_steps, loss))

    return losses_pretrain 

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
    # we don't add decay since in the original paper, there is no mention of the decay
    # scheduler = optim.lr_scheduler.StepLR(optimizer, args.nb_transitions // 3, gamma = 0.1)

    # Get initial data for pretraining
    # TODO: change here
    with open(args.demonstrations_path, 'rb') as f:
        initial_data = pickle.load(f)

    # Create replay memory
    memory = ReplayMemory(args.memory_size, initial_data, decay_proportional_dem=args.decay_proportional_dem , dem_prop_min=args.dem_prop_min, 
                            dem_prop_init=args.dem_prop_init, reward_extra=args.reward_extra, n_td=args.n_td)

    rewards_per_ep = []
    ep_reward = 0.
    losses = []

    # Pretrain
    losses_pretrain = pretrain(policy_net, target_net, memory, optimizer, args.pretrain_steps, n_actions)

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
        state = next_state.clone()

        # Update the reward
        ep_reward += reward_

        # Call the optimization function to do backprop
        loss = optimize(policy_net, target_net, optimizer, memory, scheduler=None, n_actions=n_actions)
        
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
