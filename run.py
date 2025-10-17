"""
Implementation of TAMER (Knox + Stone, 2009)
When training, use 'W' and 'A' keys for positive and negative rewards
"""

import asyncio
import gymnasium as gym

from tamer.agent import TamerRL
from pathlib import Path

MODELS_DIR = Path(__file__).parent.joinpath('tamer/saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('tamer/logs')


async def main():
    # MOUNTAINCAR_ACTION_MAP = {0: 'left', 1: 'none', 2: 'right'}
    # env = gym.make("MountainCar-v0", render_mode="rgb_array")

    CARTPOLE_ACTION_MAP = {0: 'left', 1: 'right'}
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # hyperparameters
    max_steps = 300
    discount_factor = 0.99
    epsilon = 0.1  # vanilla Q learning actually works well with no random exploration
    min_eps = 0
    num_episodes = 30
    control_sharing = True  # set false for action biasing

    # set a timestep for training TAMER
    # the more time per step, the easier for the human
    # but the longer it takes to train (in real time)
    # 0.2 seconds is fast but doable
    tamer_training_timestep = 1

    agent = TamerRL(env=env,
                    action_map=CARTPOLE_ACTION_MAP,
                    num_episodes=num_episodes,
                    max_steps=max_steps,
                    discount_factor=discount_factor,
                    epsilon=epsilon,
                    min_eps=min_eps,
                    control_sharing=control_sharing,
                    ts_len=tamer_training_timestep,
                    logs_dir=LOGS_DIR,
                    models_dir=MODELS_DIR,
                    q_model_to_load=None,
                    h_model_to_load=None)

    await agent.train(model_file_to_save=None, eval=False, eval_interval=50)
    # agent.play(n_episodes=3, render=True),
    # save_gif = True, gif_name = "500_eps_q_learning_1.gif")
    # agent.evaluate(n_episodes=30)


if __name__ == '__main__':
    asyncio.run(main())
