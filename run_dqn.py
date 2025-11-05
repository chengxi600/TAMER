import gymnasium as gym
import numpy as np
from dqn.agent import DQNAgent


def main():
    env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                   enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="rgb_array")

    agent = DQNAgent(
        env,
        action_map={0: "none", 1: "left engine",
                    2: "main engine", 3: "right engine"},
        num_episodes=5,
        max_steps=500,
        epsilon=0,
        min_epsilon=0,
        discount_factor=0.99,
        learning_rate=0.001,
        alpha_h=0,
        alpha_q=1,
        batch_size=64,
        target_update_interval=50,
        buffer_size=10000,
        ts_len=0,
        q_model_to_load=None,
        h_model_to_load=None,
        render=False,
        tamer=False)

    agent.train(
        q_model_file_to_save="dqn/saved_models/q_ll_500eps.pth",
        h_model_file_to_save="dqn/saved_models/h_ll_500eps.pth",
        eval=True,
        eval_interval=50
    )

    # agent.play(n_episodes=1, render=True, save_gif=True,
    #            gif_name="ll_dqn_300eps.gif")


if __name__ == '__main__':
    main()
