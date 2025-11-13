import gymnasium as gym
import numpy as np
from BlackJackObservationWrapper import BlackjackContinuousWrapper
from OneHotObservationWrapper import OneHotObservationWrapper

mc_env = gym.make("MountainCar-v0", render_mode="rgb_array")
mc_action_map = {0: "left", 1: "none", 2: "right"}

ll_env = gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                  enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="rgb_array")
lunar_action_map = {0: "none", 1: "right engine",
                    2: "main engine", 3: "left engine"}

racing_env = gym.make(
    "CarRacing-v3", render_mode="rgb_array", lap_complete_percent=0.95, domain_randomize=False, continuous=False)
racing_action_map = {0: 'none', 1: 'steer right',
                     2: 'steer left', 3: 'gas', 4: 'brake'}

bj_env = BlackjackContinuousWrapper(
    gym.make("Blackjack-v1", render_mode="rgb_array"))
bj_action_map = {0: 'Stand', 1: "Hit"}

lake_env = gym.make("FrozenLake-v1", render_mode="rgb_array")

print(mc_env.reset())
print(lake_env.reset())
