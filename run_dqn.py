import gymnasium as gym
import numpy as np
from BlackJackObservationWrapper import BlackjackContinuousWrapper
from OneHotObservationWrapper import OneHotObservationWrapper
from dqn.agent import DQNAgent
from dqn.configs import HYPERPARAMS
from logger import Logger
import os


def main():
    # lake_env = OneHotObservationWrapper(gym.make(
    #     'FrozenLake-v1',
    #     desc=None,
    #     map_name="4x4",
    #     is_slippery=False,
    #     reward_schedule=(100, -50, -1),
    #     render_mode="rgb_array"))
    lake_action_map = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}

    # tamer log path
    tamer_log_path = "dqn/logs/tamer/step.csv"
    # episode log path
    episode_log_path = "dqn/logs/episode/eps.csv"

    logger = Logger(episode_log_path, tamer_log_path,
                    log_csv=True, log_to_db=True)

    agent = DQNAgent(
        **HYPERPARAMS['LunarLander-v3'],
        num_episodes=10,
        ts_len=0.5,
        logger=logger,
        q_model_to_load="dqn/saved_models/q_ll_20eps.pth",
        h_model_to_load=None,
        gif_name="ll_dqntamer_10eps.gif",
        render=True,
        tamer=True,
        random_seed=2025)

    agent.train(
        name="LL 300 eps",
        q_model_file_to_save=None,
        h_model_file_to_save=None,
        eval=True,
        eval_interval=2)

    # agent.play(n_episodes=1, render=True, save_gif=True,
    #            gif_name="ll_dqn_300eps.gif")


if __name__ == '__main__':
    main()
