"""
Implementation of Deep Q Learning modified from 
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

from collections import deque, namedtuple
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import count
import torch.optim as optim


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


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


class DQNAgent:
    def __init__(self, env, num_episodes, epsilon, min_epsilon, discount_factor, learning_rate, batch_size, target_update_interval, buffer_size):
        self.env = env
        self.num_episodes = num_episodes
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.target_update_interval = target_update_interval

        self.epsilon = epsilon
        self.min_epsilon = min_epsilon
        self.epsilon_decay_step = (epsilon - min_epsilon) / num_episodes

        # initialize networks
        self.Q = QNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n)
        self.target_Q = QNetwork(
            self.env.observation_space.shape[0], self.env.action_space.n)
        self.target_Q.load_state_dict(self.Q.state_dict())

        # initialize optimizer and memory
        self.optimizer = optim.AdamW(
            self.Q.parameters(), lr=self.learning_rate, amsgrad=True)
        self.memory = ReplayBuffer(buffer_size)

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

    def train(self):
        for i in range(self.num_episodes):
            total_reward = self._train_episode()
            print(f"Episode: {i}, Reward: {total_reward}")
        self.env.close()

    def _train_episode(self):
        state, _ = self.env.reset()
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        total_reward = 0

        for ts in count():
            self.env.render()

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
                return total_reward
