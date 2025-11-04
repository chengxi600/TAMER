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
        num_episodes=50,
        epsilon=1,
        min_epsilon=0.1,
        discount_factor=0.99,
        learning_rate=0.01,
        batch_size=128,
        target_update_interval=50,
        buffer_size=10000,
        q_model_to_load=None)

    agent.train(
        model_file_to_save="dqn/saved_models/ll_5eps.pth",
        eval=False,
        eval_interval=1
    )


if __name__ == '__main__':
    main()
