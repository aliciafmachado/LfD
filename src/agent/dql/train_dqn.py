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

# from collections import namedtuple, deque
from PIL import Image

parser = argparse.ArgumentParser(description='DQN training')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR')
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


resize = T.Compose([T.ToTensor()])


# def get_screen(env):
#     # We put the image in torch order
#     screen = env.resert

#     # Convert to float, rescale, convert to torch tensor
#     # (this doesn't require a copy)
#     screen = np.ascontiguousarray(screen, dtype=np.float32) / 255

#     screen = torch.from_numpy(screen)
#     # Resize, and add a batch dimension (BCHW)
#     return resize(screen).unsqueeze(0)


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
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()

    # Compute the expected Q values using the Bellman equation
    expected_state_action_values = (next_state_values * args.gamma) + reward_batch

    # Compute huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    # TODO: do we need to clip gradients?
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

    return loss.detach().cpu().item()


def train():
    # Create environment
    env = gym.make(args.env)
    if args.state_bonus:
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

    running_reward = 0.
    ep_reward = 0.
    losses = []

    # We just take the index of objects for now
    state = torch.from_numpy(env.reset()['image'][:,:,0:1]).to(device).unsqueeze(0)

    for step in range(args.nb_transitions):

        action = policy_net.select_action(step, state, nb_transitions=args.nb_transitions)
        temp_state, reward_, done, _ = env.step(action.item())
        next_state = torch.from_numpy(temp_state['image'][:,:,0:1]).to(device).unsqueeze(0)

        reward = torch.tensor([reward_], device=device)

        memory.push(state, action, next_state, reward)
            
        # Move to next state
        state = next_state

        # Update the reward
        ep_reward += reward_

        # Call the optimization function to do backprop
        loss = optimize(policy_net, target_net, optimizer, memory)
        
        if loss is not None:
            losses.append(loss)

        if args.render:
            env.render()

        if done: 
            state = torch.from_numpy(env.reset()['image'][:,:,0:1]).to(device).unsqueeze(0)
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            ep_reward = 0

        if step % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if step % args.log_interval == 0:
            mean_loss = 0
            if len(losses) > 0:
                mean_loss = np.mean(losses)

            print('Step {}\tLast reward: {:.2f}\tAverage reward: {:.2f}\tLoss: {:.3f}'.format(
                step, reward_, running_reward, mean_loss))

    print("Finished training")
    print("Saving model to {}".format(args.save_path))
    torch.save(policy_net.state_dict(), args.save_path)


if __name__ == '__main__':
    train()

