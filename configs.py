'''
Fixed Hyperparameters for each task

'''

mountain_car_params = {
    "action_map": {0: "left", 1: "none", 2: "right"},
    "max_steps": 300,
    "discount_factor": 1,
    "epsilon": 0,
    "min_eps": 0,
    "trace_decay": 0.99,
    "trace_scaling": None,
    "trace_accum": 0.2,
}

cartpole_params = {
    "action_map": {0: 'left', 1: 'right'},
    "max_steps": 300,
    "discount_factor": 0.99,
    "epsilon": 1,
    "min_eps": 0,
    "trace_decay": 0.9,
    "trace_scaling": None,
    "trace_accum": 0.2,
}

lunar_params = {
    "action_map": {0: 'none', 1: 'left engine', 2: 'main engine', 3: 'right engine'},
    "max_steps": 1000,
    "discount_factor": 0.99,
    "epsilon": 1,
    "min_eps": 0,
    "trace_decay": 0.9,
    "trace_scaling": 1,
    "trace_accum": 0.2,
}

taxi_params = {
    "action_map": {0: 'South', 1: 'North', 2: 'East', 3: 'West', 4: 'Pickup', 5: 'Drop off'},
    "max_steps": 200,
    "discount_factor": 0.99,
    "epsilon": 1,
    "min_eps": 0,
    "trace_decay": 0.9,
    "trace_scaling": 1,
    "trace_accum": 0.2,
}

HYPERPARAMS = {
    "MountainCar-v0": mountain_car_params,
    "CartPole-v1": cartpole_params,
    "LunarLander-v3": lunar_params,
    "Taxi-v3": taxi_params,
}
