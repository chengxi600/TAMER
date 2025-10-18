"""
Implementation of TAMER (Knox + Stone, 2009)
When training, use 'W' and 'A' keys for positive and negative rewards
"""

import asyncio
import gymnasium as gym

from tamer.agent import TamerRL
from pathlib import Path
from configs import HYPERPARAMS

MODELS_DIR = Path(__file__).parent.joinpath('tamer/saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('tamer/logs')


async def main():
    # Experiment params
    num_episodes = 500
    control_sharing = False
    tamer_timestep_length = 0

    # Domain specific params
    # MountainCar
    mc_env = gym.make("MountainCar-v0", render_mode="rgb_array")
    mc_params = HYPERPARAMS["MountainCar-v0"]
    mc_params["trace_scaling"] = 2 if control_sharing else 100

    mc_agent_config = {
        **mc_params,
        "env": mc_env,
        "num_episodes": num_episodes,
        "control_sharing": control_sharing,
        "ts_len": tamer_timestep_length,
        "logs_dir": LOGS_DIR,
        "models_dir": MODELS_DIR,
        "q_model_to_load": None,
        "h_model_to_load": None
    }

    # CartPole
    cp_env = gym.make("CartPole-v1", render_mode="rgb_array")
    cp_params = HYPERPARAMS["CartPole-v1"]
    cp_params["trace_scaling"] = 1 if control_sharing else 200

    cp_agent_config = {
        **cp_params,
        "env": cp_env,
        "num_episodes": num_episodes,
        "control_sharing": control_sharing,
        "ts_len": tamer_timestep_length,
        "logs_dir": LOGS_DIR,
        "models_dir": MODELS_DIR,
        "q_model_to_load": None,
        "h_model_to_load": None
    }

    agent = TamerRL(**cp_agent_config)

    # await agent.train(model_file_to_save="500ep_cartpole_disc0.99.p", eval=True, eval_interval=50)
    agent.play(n_episodes=3, render=True, save_gif=True,
               gif_name="500ep_cartpole_disc0.99.gif")
    # agent.evaluate(n_episodes=30)


if __name__ == '__main__':
    asyncio.run(main())
