import gymnasium as gym
import numpy as np
from BlackJackObservationWrapper import BlackjackContinuousWrapper
from OneHotObservationWrapper import OneHotObservationWrapper
from dqn.agent import DQNAgent


def main():
    ll_env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                      enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="rgb_array")
    lunar_action_map = {0: "none", 1: "right engine",
                        2: "main engine", 3: "left engine"}

    taxi_env = OneHotObservationWrapper(
        gym.make("Taxi-v3", render_mode="rgb_array"))
    taxi_action_map = {0: 'South', 1: 'North',
                       2: 'East', 3: 'West', 4: 'Pickup', 5: 'Drop off'}

    mc_env = gym.make("MountainCar-v0", render_mode="rgb_array")
    mc_action_map = {0: "left", 1: "none", 2: "right"}

    cliff_env = OneHotObservationWrapper(
        gym.make("CliffWalking-v0", render_mode="rgb_array"))
    cliff_action_map = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

    bj_env = BlackjackContinuousWrapper(gym.make(
        "Blackjack-v1", natural=True, render_mode="rgb_array"))
    bj_action_map = {0: 'Stand', 1: "Hit"}

    # lake_env = OneHotObservationWrapper(gym.make(
    #     'FrozenLake-v1',
    #     desc=None,
    #     map_name="4x4",
    #     is_slippery=False,
    #     reward_schedule=(100, -50, -1),
    #     render_mode="rgb_array"))
    lake_action_map = {0: 'left', 1: 'down', 2: 'right', 3: 'up'}

    agent = DQNAgent(
        cliff_env,
        action_map=cliff_action_map,
        num_episodes=300,
        max_steps=None,
        epsilon=1,
        min_epsilon=0.1,
        discount_factor=0.99,
        learning_rate=0.001,
        alpha_h=0,
        alpha_q=1,
        alpha_h_decay=0.9999,
        batch_size=64,
        target_update_interval=20,
        buffer_size=10000,
        ts_len=0,
        q_model_to_load=None,
        h_model_to_load=None,
        gif_name="cliff_dqn_300eps.gif",
        render=False,
        tamer=False)

    agent.train(
        name="DQN Cliff Walking 300 eps",
        q_model_file_to_save=None,
        h_model_file_to_save=None,
        eval=False,
        eval_interval=50
    )

    # agent.play(n_episodes=1, render=True, save_gif=True,
    #            gif_name="ll_dqn_300eps.gif")


if __name__ == '__main__':
    main()
