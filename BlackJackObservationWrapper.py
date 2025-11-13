import gymnasium as gym
import numpy as np
import torch


class BlackjackContinuousWrapper(gym.ObservationWrapper):
    """
    Converts (player_sum, dealer_card, usable_ace) into a normalized continuous tensor.
    """

    def __init__(self, env):
        super().__init__(env)
        # Define normalization ranges
        self.low = np.array([4, 1, 0], dtype=np.float32)
        self.high = np.array([21, 10, 1], dtype=np.float32)

        # Update observation space to continuous
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(3,), dtype=np.float32
        )

    def observation(self, obs):
        player_sum, dealer_card, usable_ace = obs
        vec = np.array(
            [player_sum, dealer_card, float(usable_ace)], dtype=np.float32)
        norm_vec = (vec - self.low) / (self.high - self.low)
        # shape (1, 3)
        return np.array(norm_vec, dtype=np.float32)
