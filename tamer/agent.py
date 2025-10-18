import datetime as dt
import os
import pickle
import time
import uuid
from itertools import count
from pathlib import Path
from sys import stdout
from csv import DictWriter
import imageio

import numpy as np
from sklearn import pipeline, preprocessing
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor

from .interface import Interface
from .logger import Logger

MOUNTAINCAR_ACTION_MAP = {0: 'left', 1: 'none', 2: 'right'}
CARTPOLE_ACTION_MAP = {0: 'left', 1: 'right'}
MODELS_DIR = Path(__file__).parent.joinpath('saved_models')
LOGS_DIR = Path(__file__).parent.joinpath('logs')


class SGDFunctionApproximator:
    """ SGD function approximator with RBF preprocessing. """

    def __init__(self, env):

        # Feature preprocessing: Normalize to zero mean and unit variance
        # We use a few samples from the observation space to do this
        observation_examples = np.array(
            [env.observation_space.sample() for _ in range(10000)], dtype='float64'
        )
        self.scaler = preprocessing.StandardScaler()
        self.scaler.fit(observation_examples)

        # Used to convert a state to a featurized represenation.
        # We use RBF kernels with different variances to cover different parts of the space
        self.featurizer = pipeline.FeatureUnion(
            [
                ('rbf1', RBFSampler(gamma=5.0, n_components=100)),
                ('rbf2', RBFSampler(gamma=2.0, n_components=100)),
                ('rbf3', RBFSampler(gamma=1.0, n_components=100)),
                ('rbf4', RBFSampler(gamma=0.5, n_components=100)),
            ]
        )
        self.featurizer.fit(self.scaler.transform(observation_examples))

        self.models = []
        for _ in range(env.action_space.n):
            model = SGDRegressor(learning_rate='constant')
            obs, _ = env.reset()
            model.partial_fit([self.featurize_state(obs)], [0])
            self.models.append(model)

    def predict(self, state, action=None):
        features = self.featurize_state(state)
        if not action:
            return [m.predict([features])[0] for m in self.models]
        else:
            return self.models[action].predict([features])[0]

    def update(self, state, action, td_target):
        features = self.featurize_state(state)
        self.models[action].partial_fit([features], [td_target])

    def featurize_state(self, state):
        """ Returns the featurized representation for a state. """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]


class Tamer:
    """
    QLearning Agent adapted to TAMER using steps from:
    http://www.cs.utexas.edu/users/bradknox/kcap09/Knox_and_Stone,_K-CAP_2009.html
    """

    def __init__(
        self,
        env,
        num_episodes,
        max_steps,  # max timesteps for training
        discount_factor=1,  # only affects Q-learning
        epsilon=0,  # only affects Q-learning
        min_eps=0,  # minimum value for epsilon after annealing
        tame=True,  # set to false for normal Q-learning
        ts_len=0.2,  # length of timestep for training TAMER
        output_dir=LOGS_DIR,
        model_file_to_load=None  # filename of pretrained model
    ):
        self.tame = tame
        self.ts_len = ts_len
        self.env = env
        self.uuid = uuid.uuid4()
        self.output_dir = output_dir
        self.max_steps = max_steps

        # init model
        if model_file_to_load is not None:
            print(f'Loaded pretrained model: {model_file_to_load}')
            self.load_model(filename=model_file_to_load)
        else:
            if tame:
                self.H = SGDFunctionApproximator(env)  # init H function
            else:  # optionally run as standard Q Learning
                self.Q = SGDFunctionApproximator(env)  # init Q function

        # hyperparameters
        self.discount_factor = discount_factor
        self.epsilon = epsilon if not tame else 0
        self.num_episodes = num_episodes
        self.min_eps = min_eps

        # calculate episodic reduction in epsilon
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # tamer log path
        tamer_log_path = os.path.join(
            self.output_dir, "tamer", f'{self.uuid}.csv')
        # episode log path
        episode_log_path = os.path.join(
            self.output_dir, "episode", f'{self.uuid}.csv')

        # Logger
        self.logger = Logger(episode_log_path, tamer_log_path)

    def act(self, state):
        """ Epsilon-greedy Policy """
        if np.random.random() < 1 - self.epsilon:
            preds = self.H.predict(
                state) if self.tame else self.Q.predict(state)
            return np.argmax(preds)
        else:
            return np.random.randint(0, self.env.action_space.n)

    def _train_episode(self, episode_index, disp: Interface):
        print(f'Episode: {episode_index + 1}')
        tot_reward = 0
        state, _ = self.env.reset()
        ep_start_time = dt.datetime.now().time()
        for ts in count():
            frame = self.env.render()

            # Determine next action
            action = self.act(state)
            disp.render(frame, action)

            # Get next state and reward
            next_state, reward, done, truncated, info = self.env.step(
                action)

            if not self.tame:
                if done and next_state[0] >= 0.5:
                    td_target = reward
                else:
                    td_target = reward + self.discount_factor * np.max(
                        self.Q.predict(next_state)
                    )
                self.Q.update(state, action, td_target)
            else:
                pass
                now = time.time()
                while time.time() < now + self.ts_len:
                    frame = None

                    time.sleep(0.01)  # save the CPU

                    human_reward = disp.get_scalar_feedback()
                    feedback_ts = dt.datetime.now().time()
                    if human_reward != 0:
                        self.logger.log_tamer_step(
                            episode_index, ep_start_time, feedback_ts, human_reward, reward)
                        self.H.update(state, action, human_reward)
                        break

            tot_reward += reward
            if done or ts >= self.max_steps-1:
                print(f'Reward: {tot_reward}')
                ep_end_time = dt.datetime.now().time()
                return ep_start_time, ep_end_time, tot_reward

            stdout.write('\b' * (len(str(ts)) + 1))
            state = next_state
        print(f'Steps: {ts}')
        # Decay epsilon
        if self.epsilon > self.min_eps:
            self.epsilon -= self.epsilon_step
        print("-----------------------")

    async def train(self, model_file_to_save=None, eval=False, eval_interval=1):
        """
        TAMER (or Q learning) training loop
        Args:
            model_file_to_save: save Q or H model to this filename
        """
        self.env.reset()
        disp = Interface(action_map=CARTPOLE_ACTION_MAP,
                         env_frame_shape=self.env.render().shape, tamer=self.tame)
        for i in range(self.num_episodes):
            ep_start_time, ep_end_time, total_reward = self._train_episode(
                i, disp)
            self.logger.log_episode(
                i, ep_start_time, ep_end_time, total_reward)
            if eval and i > 1 and (i+1) % eval_interval == 0:
                avg_reward = self.evaluate(n_episodes=30)
                self.logger.log_episode(
                    i, "eval", "eval", avg_reward)

        print('\nCleaning up...')
        disp.close()
        if model_file_to_save is not None:
            print(f'\nSaving Model to {model_file_to_save}')
            self.save_model(filename=model_file_to_save)

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
        if render:
            disp = Interface(action_map=CARTPOLE_ACTION_MAP,
                             env_frame_shape=self.env.render().shape, tamer=False)
        for i in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            tot_reward = 0
            for i in count():
                action = self.act(state)
                next_state, reward, done, truncated, info = self.env.step(
                    action)
                tot_reward += reward
                frames.append(self.env.render())
                if render:
                    disp.render(self.env.render(), action)
                if i >= self.max_steps-1 or done:
                    break
                state = next_state
            ep_rewards.append(tot_reward)
            # print(f'Episode: {i + 1} Reward: {tot_reward}')
        if render:
            disp.close()

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

    def save_model(self, filename):
        """
        Save H or Q model to models dir
        Args:
            filename: name of pickled file
        """
        model = self.H if self.tame else self.Q
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, filename):
        """
        Load H or Q model from models dir
        Args:
            filename: name of pickled file
        """
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(MODELS_DIR.joinpath(filename), 'rb') as f:
            model = pickle.load(f)
        if self.tame:
            self.H = model
        else:
            self.Q = model
