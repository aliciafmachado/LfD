"""
Classic Replay Buffer.
"""

import random
from src.utils.transition import Transition, TransitionTD
from collections import deque


class ReplayMemory(object):
    def __init__(self, capacity=100000, initial_data={}, dem_prop_init=0., decay_proportional_dem=0., dem_prop_min=0., n_td=4, gamma=0.99, 
                 reward_extra=False, reward_increase=0.25, decay_e_reward=0.5):
        super(ReplayMemory, self).__init__()
        self.memory = deque([], maxlen=capacity)
        self.dem_memory = deque([], maxlen=capacity)
        self.capacity = capacity

        self.n_td = n_td
        self.gamma = gamma
        self.reward_extra = reward_extra
        self.reward_increase = reward_increase
        self.decay_e_reward = decay_e_reward

        for trajectory in initial_data:
            self._push_trajectory(self.dem_memory, trajectory)
        
        self.use_demonstrations = len(self.dem_memory) > 0
        self.dem_prop_init = dem_prop_init
        self.dem_prop_min = dem_prop_min
        self.decay_proportional_dem = decay_proportional_dem

    def push_trajectory(self, trajectory):
        self._push_trajectory(self.memory, trajectory)

    def _push_trajectory(self, mem, trajectory):
        """
        Trajectory is a dict with different values.
        """
        # Calculate TD error and next actions
        len_traj = len(trajectory['reward'])

        for i, tr in enumerate(zip(*trajectory.values())):
            trans = Transition(*tr)
            R_cum = [(self.gamma ** j) * trajectory['reward'][i + j] for j in range(min(self.n_td, len_traj - i))]
            n_next_state = trajectory['state'][min(i + self.n_td, len_traj - 1)]
            gamma_n = len(R_cum)
            
            R_cum = sum(R_cum)

            trans_td = TransitionTD(*trans, n_next_state=n_next_state, R_cum=R_cum, gamma_n=gamma_n)
            mem.append(trans_td)

    def get(self, idx, mem='demo'):
        return self.memory[idx] if mem == 'memory' else self.dem_memory[idx]

    def sample_demonstrations(self, batch_size):
        return random.sample(self.dem_memory, batch_size)

    def sample_memory(self, batch_size):
        return random.sample(self.memory, batch_size)

    def sample(self, batch_size):
        def mult_reward(tr, incr):
            return TransitionTD(state=tr.state, action=tr.action, reward=tr.reward + incr, 
                                next_state=tr.next_state, n_next_state=tr.n_next_state, 
                                R_cum=tr.R_cum + incr * (sum([self.gamma ** i for i in range(tr.gamma_n)])), 
                                gamma_n=tr.gamma_n)
        # Get dem factor
        if self.decay_proportional_dem > 1e-5:
            dem_factor = self.dem_prop_min + (self.dem_prop_init - self.dem_prop_min) * max(
                0, 1 - (len(self.memory) / (self.decay_proportional_dem * self.capacity)))
        else:
            dem_factor = self.dem_prop_init

        # We return the samples and their indexes so that
        # we are able to retrieve the td error and next action
        if self.use_demonstrations:
            batch_dem = int(dem_factor * batch_size)
        else:
            batch_dem = 0

        # If we don't have enough data available to follow the proportions
        # we use only what we have and complete with demonstrations
        if batch_size - batch_dem > len(self.memory): 
            batch_samp = len(self.memory)
            batch_dem = batch_size - batch_samp
        
        else:
            batch_samp = batch_size - batch_dem

        sample_dem = self.sample_demonstrations(batch_dem)
        sample_mem = self.sample_memory(batch_samp)

        # Add extra reward to the demonstrations
        if self.reward_extra:
            r = self.reward_increase * max(0, 1 - (len(self.memory) / (self.decay_e_reward * self.capacity)))
            # Fix R_accum as well
            sample_dem = [mult_reward(tr, r) for tr in sample_dem]
            
        return sample_dem + sample_mem

    def __len__(self):
        return len(self.memory) + len(self.dem_memory)