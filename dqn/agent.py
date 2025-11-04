"""
Implementation of Deep Q Learning modified from
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

from collections import deque, namedtuple
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
import torch.optim as optim
import os
import imageio
import datetime as dt
import uuid
from logger import Logger
from interface import Interface


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
LOGS_DIR = Path(__file__).parent.joinpath('logs')


class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.layer1 = nn.Linear(state_size, 128)
        self.layer2 = nn.Linear(128, 128)
        self.out_layer = nn.Linear(128, action_size)

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
        epsilon,
        min_epsilon,
        discount_factor,
        learning_rate,
        batch_size,
        target_update_interval,
        buffer_size,
        logs_dir=LOGS_DIR,
        q_model_to_load=None,  # filename of pretrained Q model
    ):
        self.env = env
        self.env.reset()
        self.num_episodes = num_episodes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval
        self.logs_dir = logs_dir
        self.uuid = uuid.uuid4()
        self.disp = Interface(action_map=action_map,
                              env_frame_shape=self.env.render().shape)

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_step = (epsilon - min_epsilon) / num_episodes

        # initialize networks
        self.Q = QNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n)
        if q_model_to_load is not None:
            self.Q.load_state_dict(torch.load(
                q_model_to_load, weights_only=True))

        self.target_Q = QNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n)
        self.target_Q.load_state_dict(self.Q.state_dict())

        # initialize optimizer and memory
        self.optimizer = optim.AdamW(
            self.Q.parameters(), lr=self.learning_rate, amsgrad=True)
        self.memory = ReplayBuffer(buffer_size)

        # tamer log path
        tamer_log_path = os.path.join(
            self.logs_dir, "tamer", f'{self.uuid}.csv')
        # episode log path
        episode_log_path = os.path.join(
            self.logs_dir, "episode", f'{self.uuid}.csv')

        # Logger
        self.logger = Logger(episode_log_path, tamer_log_path)

    def act(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                return self.Q(state).max(1).indices.view(1, 1)
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

    def train(self, model_file_to_save=None, eval=False, eval_interval=1):
        """
        TAMER (or Q learning) training loop
        Args:
            model_file_to_save: save Q or H model to this filename
        """
        self.env.reset()
        for i in range(self.num_episodes):
            ep_start_time, ep_end_time, total_reward = self._train_episode()
            print(f'Episode {i}, Reward: {total_reward}')
            self.logger.log_episode(
                i, ep_start_time, ep_end_time, total_reward)
            if eval and i > 1 and (i+1) % eval_interval == 0:
                avg_reward = self.evaluate(n_episodes=30)
                self.logger.log_episode(
                    i, "eval", "eval", avg_reward)

                print('\nCleaning up...')
        self.env.close()
        if model_file_to_save is not None:
            print(f'\nSaving Q Model to {model_file_to_save}')
            self.Q.save_model(model_file_to_save)

    def _train_episode(self):
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
            done = truncated or terminated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    observation, dtype=torch.float32).unsqueeze(0)

            self.disp.render(self.env.render(), action.item())

            # Store transition in replay buffer
            self.memory.push(state, action, next_state, reward)

            # Move to next state
            state = next_state

            # Gradient descent
            self.learn()

            # Every C steps, update the target network
            if ts % self.target_update_interval == 0:
                self.target_Q.load_state_dict(self.Q.state_dict())

            # Decay epsilon
            if self.epsilon > self.min_epsilon:
                self.epsilon -= self.epsilon_decay_step

            if done:
                ep_end_time = dt.datetime.now().time()
                return ep_start_time, ep_end_time, total_reward

    def play(self, n_episodes=1, render=False, save_gif=False, gif_name="agent.gif"):
        """
            Run episodes with trained agent
            Args:
                n_episodes: number of episodes
                render: optionally render episodes

            Returns: list of cumulative episode rewards
            """
        self.epsilon = 0
        ep_rewards = []
        frames = []
        self.env.reset()
        for i in range(n_episodes):
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            done = False
            tot_reward = 0
            for i in count():
                action = self.act(state)
                observation, reward, terminated, truncated, info = self.env.step(
                    action.item())
                done = truncated or terminated
                next_state = torch.tensor(
                    observation, dtype=torch.float32).unsqueeze(0)
                tot_reward += reward
                frames.append(self.env.render())

                if render:
                    self.disp.render(self.env.render(), action.item())
                if i >= self.max_steps-1 or done:
                    break
                state = next_state
            ep_rewards.append(tot_reward)
        if render:
            self.env.close()

        # only saves gif of the last run
        if save_gif:
            imageio.mimsave(gif_name, frames, fps=30)
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
