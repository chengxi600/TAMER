"""
Implementation of TAMER (Knox + Stone, 2009)
When training, use 'W' and 'A' keys for positive and negative rewards
"""

import asyncio
import gymnasium as gym

from tamer.agent import Tamer


async def main():
    env = gym.make("CartPole-v1", render_mode="rgb_array")

    # hyperparameters
    max_steps = 300
    discount_factor = 1
    epsilon = 0  # vanilla Q learning actually works well with no random exploration
    min_eps = 0
    num_episodes = 10
    tame = True  # set to false for vanilla Q learning

    # set a timestep for training TAMER
    # the more time per step, the easier for the human
    # but the longer it takes to train (in real time)
    # 0.2 seconds is fast but doable
    tamer_training_timestep = 0.5

    agent = Tamer(env, num_episodes, max_steps, discount_factor, epsilon, min_eps, tame,
                  tamer_training_timestep, model_file_to_load="cp_10ep_tamer.p")

    # await agent.train(model_file_to_save='cp_10ep_tamer.p', eval=True, eval_interval=2)
    agent.play(n_episodes=3, render=False, save_gif=True,
               gif_name="cp_10ep_tamer.gif")
    # agent.evaluate(n_episodes=30)


if __name__ == '__main__':
    asyncio.run(main())
