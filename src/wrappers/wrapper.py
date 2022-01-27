"""
Wrapper to return framestack.
"""

import numpy as np
import gym


NB_FRAMES = 4


def preprocess(state):
    return state['image'][:,:,0:1]

class FrameStack(gym.core.Wrapper):
    """
    Returns FrameStack object.
    """

    def __init__(self, env):
        super().__init__(env)
        self.stack_states = None

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.stack_states = np.concatenate([self.stack_states[:,:,1:], preprocess(obs)], axis=2)
        return self.stack_states, reward, done, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self.stack_states = np.tile(preprocess(obs), NB_FRAMES)
        return self.stack_states