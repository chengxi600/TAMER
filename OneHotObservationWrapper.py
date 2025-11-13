import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class OneHotObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Discrete):
            raise Exception(
                'This wrapper is only for environments with Discrete observation space')

        self.base_observation_space = env.observation_space
        num_categories = self.base_observation_space.n
        self.observation_space = Box(
            low=0, high=1, shape=(num_categories,), dtype=np.float32)

    def observation(self, observation):
        one_hot_obs = np.zeros(
            self.base_observation_space.n, dtype=np.float32)
        one_hot_obs[observation] = 1.0
        return one_hot_obs
