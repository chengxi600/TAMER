import gymnasium as gym

from BlackJackObservationWrapper import BlackjackContinuousWrapper
from OneHotObservationWrapper import OneHotObservationWrapper

mountain_car_params = {
    "env": gym.make("MountainCar-v0", render_mode="rgb_array"),
    "action_map": {0: "left", 1: "none", 2: "right"},
    "max_steps": 1000,
    "discount_factor": 1,
    "epsilon": 0,
    "min_epsilon": 0,
    "learning_rate": 0.001,
    "alpha_h": 0,
    "alpha_q": 1,
    "alpha_h_decay": 0.9999,
    "batch_size": 64,
    "target_update_interval": 20,
    "buffer_size": 10000,
}

lunar_params = {
    "env": gym.make("LunarLander-v3", continuous=False, gravity=-10.0,
                    enable_wind=False, wind_power=15.0, turbulence_power=1.5, render_mode="rgb_array"),
    "action_map": {0: "none", 1: "right engine", 2: "main engine", 3: "left engine"},
    "max_steps": 500,
    "discount_factor": 0.97,
    "epsilon": 1,
    "min_epsilon": 0,
    "learning_rate": 0.001,
    "alpha_h": 1,
    "alpha_q": 1,
    "alpha_h_decay": 0.9999,
    "batch_size": 64,
    "target_update_interval": 20,
    "buffer_size": 3000,
}

taxi_params = {
    "env": OneHotObservationWrapper(gym.make("Taxi-v3", render_mode="rgb_array")),
    "action_map": {0: 'South', 1: 'North', 2: 'East', 3: 'West', 4: 'Pickup', 5: 'Drop off'},
    "max_steps": 500,
    "discount_factor": 0.97,
    "epsilon": 1,
    "min_epsilon": 0,
    "learning_rate": 0.001,
    "alpha_h": 1,
    "alpha_q": 1,
    "alpha_h_decay": 0.9999,
    "batch_size": 64,
    "target_update_interval": 20,
    "buffer_size": 1000,
}

cliff_params = {
    "env": OneHotObservationWrapper(gym.make("CliffWalking-v0", render_mode="rgb_array")),
    "action_map": {0: 'up', 1: 'right', 2: 'down', 3: 'left'},
    "max_steps": 300,
    "discount_factor": 0.97,
    "epsilon": 1,
    "min_epsilon": 0,
    "learning_rate": 0.001,
    "alpha_h": 1,
    "alpha_q": 1,
    "alpha_h_decay": 0.9999,
    "batch_size": 64,
    "target_update_interval": 20,
    "buffer_size": 1000,
}

bj_params = {
    "env": BlackjackContinuousWrapper(gym.make("Blackjack-v1", natural=True, render_mode="rgb_array")),
    "action_map": {0: 'Stand', 1: "Hit"},
    "max_steps": None,
    "discount_factor": 1,
    "epsilon": 1,
    "min_epsilon": 0,
    "learning_rate": 0.001,
    "alpha_h": 0,
    "alpha_q": 1,
    "alpha_h_decay": 0.9999,
    "batch_size": 64,
    "target_update_interval": 20,
    "buffer_size": 10000,
}

HYPERPARAMS = {
    "MountainCar-v0": mountain_car_params,
    "LunarLander-v3": lunar_params,
    "Taxi-v3": taxi_params,
    "CliffWalking-v0": cliff_params,
    "Blackjack-v1": bj_params,
}
