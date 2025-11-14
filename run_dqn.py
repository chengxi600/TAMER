import gymnasium as gym
import numpy as np
from BlackJackObservationWrapper import BlackjackContinuousWrapper
from OneHotObservationWrapper import OneHotObservationWrapper
from dqn.agent import DQNAgent
from dqn.configs import HYPERPARAMS


def main():
    # lake_env = OneHotObservationWrapper(gym.make(
    #     'FrozenLake-v1',
    #     desc=None,
    #     map_name="4x4",
    #     is_slippery=False,
    #     reward_schedule=(100, -50, -1),
    #     render_mode="rgb_array"))
    lake_action_map = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}

    agent = DQNAgent(
        **HYPERPARAMS['LunarLander-v3'],
        num_episodes=20,
        ts_len=0.3,
        q_model_to_load=None,
        h_model_to_load=None,
        gif_name="dqn-tamer_ll_20eps.gif",
        render=True,
        tamer=True)

    agent.train(
        name="DQN-TAMER Lunar Lander 20 eps",
        q_model_file_to_save="q_dqntamer_ll_20eps.pth",
        h_model_file_to_save="h_dqntamer_ll_20eps.pth",
        eval=True,
        eval_interval=2
    )

    # agent.play(n_episodes=1, render=True, save_gif=True,
    #            gif_name="ll_dqn_300eps.gif")


if __name__ == '__main__':
    main()
