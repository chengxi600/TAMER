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
    "discount_factor": 0.95,
    "epsilon": 0,
    "min_eps": 0,
    "trace_decay": 0.9,
    "trace_scaling": None,
    "trace_accum": 0.2,
}

HYPERPARAMS = {
    "MountainCar-v0": mountain_car_params,
    "CartPole-v1": cartpole_params,
}
