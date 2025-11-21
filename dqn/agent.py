"""
Implementation of Deep Q Learning modified from
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
import torch.optim as optim
import time
import os
import imageio
import datetime as dt
import uuid
from logger import Logger
from interface import Interface
from dqn.replay import ReplayBuffer, HumanTransition, Transition
from datetime import datetime


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.out_layer = nn.Linear(128, output_size)

        nn.init.zeros_(self.out_layer.weight)
        nn.init.zeros_(self.out_layer.bias)

    def forward(self, state):
        x = self.layer1(state)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        return self.out_layer(x)

    def save_model(self, filepath):
        """
        Saves a model to models dir
        Args:
            filename: name of pickled file
        """
        torch.save(self.state_dict(), filepath)


class DQNAgent:
    def __init__(
        self,
        env,
        action_map,
        num_episodes,
        max_steps,
        epsilon,
        min_epsilon,
        alpha_q,
        alpha_h,
        alpha_h_decay,
        discount_factor,
        learning_rate,
        batch_size,
        target_update_interval,
        buffer_size,
        ts_len,
        logger,
        q_model_to_load=None,  # filename of pretrained Q model
        h_model_to_load=None,  # filename of pretrained H model
        gif_name="agent.gif",
        render=True,
        tamer=False,
    ):
        """ Initializes a DQN Agent

        Args:
            env (gymnasium.Env): a gym environment
            action_map (dict[int, str]): a dict that maps an action to the description of action
            num_episodes (int): number of training episodes
            max_steps (int): max steps per episode
            epsilon (float): exploration rate
            min_epsilon (float): minimum exploration rate (epsilon will be decayed to min_epsilon)
            alpha_q (float): action biasing weight for Q function
            alpha_h (float): action biasing weight for H function
            alpha_h_decay (float): decay for the action biasing weight for H function
            discount_factor (float): discount factor for rewards
            learning_rate (float): learning rate for Q and H functions
            batch_size (int): learning batch size
            target_update_interval (int): frequency of updating target Q network
            buffer_size (int): buffer size for replay memory
            ts_len (float): temporal length of a timestep (seconds)
            logger (Logger): logger object for logging 
            q_model_to_load (string, optional): file to load Q model. Defaults to None.
            h_model_to_load (string, optional): file to load H model. Defaults to None.
            gif_name (str, optional): filename of saved gif of trained policy
            render (bool, optional): renders environment. Defaults to True.
            tamer (bool, optional): allow human feedback. Defaults to False.
        """
        self.env = env
        self.env.reset()
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.uuid = uuid.uuid4()
        self.ts_len = ts_len
        self.gif_name = gif_name
        self.render = render
        self.tamer = tamer
        if self.render:
            self.disp = Interface(action_map=action_map,
                                  env_frame_shape=self.env.render().shape)

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_step = (epsilon - min_epsilon) / (num_episodes/3)

        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.alpha_q = alpha_q
        self.alpha_h = alpha_h
        self.alpha_h_decay = alpha_h_decay

        # initialize networks
        # get observation space
        obs_dim = np.prod(self.env.observation_space.shape)

        # initialize Q
        self.Q = QNetwork(
            obs_dim, self.env.action_space.n)
        if q_model_to_load is not None:
            self.Q.load_state_dict(torch.load(
                q_model_to_load, weights_only=True))

        self.target_Q = QNetwork(
            obs_dim, self.env.action_space.n)
        self.target_Q.load_state_dict(self.Q.state_dict())

        # initialize H
        self.H = QNetwork(
            obs_dim, self.env.action_space.n)
        if h_model_to_load is not None:
            self.H.load_state_dict(torch.load(
                h_model_to_load, weights_only=True))

        # initialize optimizer and memory
        self.optimizer = optim.AdamW(
            self.Q.parameters(), lr=self.learning_rate, amsgrad=True)
        self.H_optimizer = optim.AdamW(
            self.H.parameters(), lr=self.learning_rate, amsgrad=True)
        self.memory = ReplayBuffer(buffer_size, Transition)
        self.human_memory = ReplayBuffer(buffer_size, HumanTransition)

        # Logger
        self.logger = logger

    def act(self, state, eval=False):
        if random.random() > self.epsilon:
            with torch.no_grad():
                q_values = self.Q(state)
                h_values = self.H(state)
                if eval:
                    combined = q_values * self.alpha_q
                else:
                    combined = q_values * self.alpha_q + h_values * self.alpha_h

                # break ties within a 1e-6 tolerance
                max_val = combined.max()
                mask = (combined - max_val).abs() < 1e-6

                # SELECT ACTION INDICES (dimension=1)
                max_actions = torch.where(mask)[1]

                action = max_actions[torch.randint(0, len(max_actions), (1,))]
                return action.view(1, 1)

        else:
            return torch.tensor([[self.env.action_space.sample()]], dtype=torch.long)

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                          if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # get Q(s_t, a) values
        state_action_values = self.Q(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_Q(
                non_final_next_states).max(1).values

        # Compute the expected Q values
        expected_state_action_values = (
            next_state_values * self.discount_factor) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values,
                         expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.Q.parameters(), 100)
        self.optimizer.step()

    def learn_human(self):
        if len(self.human_memory) < self.batch_size:
            return

        transitions = self.human_memory.sample(self.batch_size)

        # This converts batch-array of HumanTransitions to HumanTransitions of batch-arrays.
        batch = HumanTransition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        feedback_batch = torch.cat(batch.feedback)

        predicted_feedback = self.H(state_batch).gather(1, action_batch)

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(predicted_feedback,
                         feedback_batch.unsqueeze(1))

        # Optimize the model
        self.H_optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.H.parameters(), 100)
        self.H_optimizer.step()

    def train(
        self,
        name="Experiment",
        q_model_file_to_save=None,
        h_model_file_to_save=None,
        eval=False,
        eval_interval=1
    ):
        """
        TAMER (or Q learning) training loop
        Args:
            model_file_to_save: save Q or H model to this filename
        """
        self.env.reset()
        for i in range(self.num_episodes):
            ep_start_time, ep_end_time, total_reward = self._train_episode(i)
            print(f'Episode {i}, Reward: {total_reward}')
            self.logger.log_episode(
                i, ep_start_time, ep_end_time, total_reward)
            if eval and i > 1 and (i+1) % eval_interval == 0:
                avg_reward = self.evaluate(n_episodes=30)
                self.logger.log_episode(
                    i, "eval", "eval", avg_reward)

                print('\nCleaning up...')
        self.logger.csv_logger.close()

        if self.render:
            self.disp.close()
        if q_model_file_to_save is not None:
            print(f'\nSaving Q Model to {q_model_file_to_save}')
            self.Q.save_model(q_model_file_to_save)

        if h_model_file_to_save is not None:
            print(f'\nSaving H Model to {h_model_file_to_save}')
            self.H.save_model(h_model_file_to_save)

        print("\nSaving logs to database...")
        try:
            self.play(n_episodes=1, render=False,
                      save_gif=True, gif_name=self.gif_name)
            self.logger.log_gif(self.gif_name)
            self.logger.log_experiment(name=name, date=datetime.today().strftime(
                '%Y-%m-%d'), algorithm="DQN-TAMER" if self.tamer else "DQN")

        except Exception as e:
            print(f"Exception: {e}. Failed to save logs to database")

    def _train_episode(self, ep_idx):
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0
        ep_start_time = dt.datetime.now().time()

        for ts in count():
            # Determine next action
            action = self.act(state)

            # Get next state and reward
            observation, reward, terminated, truncated, info = self.env.step(
                action.item())
            total_reward += reward
            reward = torch.tensor([reward])
            done = terminated or (
                self.max_steps is not None and ts > self.max_steps) or (self.max_steps is None and truncated)

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32).unsqueeze(0)

            if self.render:
                self.disp.render(self.env.render(), action.item())

            now = time.time()
            while time.time() < now + self.ts_len:
                time.sleep(0.01)  # save the CPU
                feedback = self.disp.get_scalar_feedback()
                feedback_ts = dt.datetime.now().time()

                if feedback != 0:
                    # self.logger.log_tamer_step(
                    #     ep_idx, ts, feedback, reward)
                    self.human_memory.push(
                        state, action, torch.tensor([feedback]))

                    if self.render:
                        self.disp.render(self.env.render(), action.item())

            # Store transition in replay buffer
            self.memory.push(state, action, next_state, reward)

            # Move to next state
            state = next_state

            # Gradient descent
            self.learn()
            self.learn_human()

            # Every C steps, update the target network
            if ts % self.target_update_interval == 0:
                self.target_Q.load_state_dict(self.Q.state_dict())

            # Decay alpha_h
            self.alpha_h *= self.alpha_h_decay

            if done:
                ep_end_time = dt.datetime.now().time()
                return ep_start_time, ep_end_time, total_reward

        # Decay epsilon after an episode
        if self.epsilon > self.min_epsilon:
            self.epsilon -= self.epsilon_decay_step

    def play(self, n_episodes=1, render=False, save_gif=False, gif_name="agent.gif"):
        """
          Run episodes with trained agent
          Args:
              n_episodes: number of episodes
              render: optionally render episodes

          Returns: list of cumulative episode rewards
        """
        ep_rewards = []
        frames = []
        self.env.reset()
        for i in range(n_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            done = False
            tot_reward = 0
            for i in count():
                action = self.act(state, eval=True)
                observation, reward, terminated, truncated, info = self.env.step(
                    action.item())
                done = truncated or terminated
                next_state = torch.tensor(
                    observation, dtype=torch.float32).unsqueeze(0)
                tot_reward += reward
                frames.append(self.env.render())

                if render:
                    self.disp.render(self.env.render(), action.item())
                if done or (self.max_steps is not None and i >= self.max_steps-1):
                    break
                state = next_state
            ep_rewards.append(tot_reward)
        if render:
            self.disp.close()

        # only saves gif of the last run
        if save_gif:
            imageio.mimsave(gif_name, frames, fps=30, loop=0)
        return ep_rewards

    def evaluate(self, n_episodes=100):
        print('Evaluating agent')
        rewards = self.play(n_episodes=n_episodes)
        avg_reward = np.mean(rewards)
        print(
            f'Average total episode reward over {n_episodes} '
            f'episodes: {avg_reward:.2f}'
        )
        return avg_reward
