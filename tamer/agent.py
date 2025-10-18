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
        """ Returns the featurized representation for a state. Since RBF centers data around
            [-1,1], we shift the values to be [0,1].
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        # features = (featurized[0] + 1.0) / 2.0
        return featurized[0]


class EligibilityModule:
    def __init__(self, feature_dim, n_actions, trace_decay, trace_scaling, trace_accum, control_sharing):
        self.feature_dim = feature_dim
        self.n_actions = n_actions
        self.trace_decay = trace_decay
        self.trace_scaling = trace_scaling
        self.trace_accum = trace_accum
        self.control_sharing = control_sharing
        self.trace = np.zeros(feature_dim)

        if not control_sharing:
            self.action_traces = np.zeros((n_actions, feature_dim))

    def compute_beta(self, feature_vector, action=None):
        """
            Get beta for the current state-action. Beta is a function of a
            constant scaling factor, eligibility module, and feature vector.
        """
        if not self.control_sharing and action is None:
            raise ValueError(
                "Action must be provided for action-biasing beta computation.")

        trace = self.trace if self.control_sharing else self.action_traces[action]

        norm = np.linalg.norm(feature_vector, ord=1)
        dot_product = np.dot(trace, feature_vector)

        return self.trace_scaling * dot_product / norm

    def update_trace(self, feature_vector, action=None):
        """
            updates the eligibility module, where each trace caps at 1.
            e_i := min(1, e_i + (f_ni * a)) for all i in eligibility trace
        """
        if not self.control_sharing and action is None:
            raise ValueError(
                "Action must be provided for action-biasing trace update.")

        if self.control_sharing:
            self.trace = np.minimum(
                1, self.trace + feature_vector * self.trace_accum)
        else:
            self.action_traces[action] = np.minimum(
                1, self.action_traces[action] + feature_vector * self.trace_accum)

    def decay_trace(self, action=None):
        """ decays the eligibility trace. If given action, decays trace for all other
            traces except given action
        """
        if self.control_sharing:
            self.trace = self.trace * self.trace_decay
        elif action is None:
            self.action_traces *= self.trace_decay
        else:
            indices = [i for i in range(self.n_actions) if i != action]
            self.action_traces[indices] *= self.trace_decay

    def reset_trace(self):
        """Resets eligibility trace
        """
        self.trace = np.zeros(self.feature_dim)
        if not self.control_sharing:
            self.action_traces = np.zeros((self.n_actions, self.feature_dim))


class TamerRL:
    """
    TAMER + RL Agent adapted from
    https://bradknox.net/public/papers/aamas12-knox.pdf
    """

    def __init__(
        self,
        env,
        action_map,
        num_episodes,
        max_steps,  # max timesteps for training
        control_sharing,  # True for control sharing improvement, False for action biasing
        discount_factor=1,  # discount factor for Q-learning
        trace_decay=0.99,  # decay factor for eligibility trace
        trace_scaling=1,  # scaling factor to calculate beta
        trace_accum=1,  # accumulation speed for eligibility trace
        epsilon=0,  # exploration for action selection
        min_eps=0,  # minimum value for epsilon after annealing
        ts_len=0.2,  # length of timestep for training TAMER
        logs_dir=LOGS_DIR,  # output directory for logs
        models_dir=MODELS_DIR,  # output directory for models
        q_model_to_load=None,  # filename of pretrained Q model
        h_model_to_load=None  # filename of pretrained H model
    ):
        self.ts_len = ts_len
        self.env = env
        self.uuid = uuid.uuid4()
        self.logs_dir = logs_dir
        self.models_dir = models_dir
        self.max_steps = max_steps
        self.action_map = action_map
        self.control_sharing = control_sharing

        # init model
        if q_model_to_load is not None:
            print(f'Loaded pretrained Q model: {q_model_to_load}')
            self.load_model(filename=q_model_to_load)
        else:
            self.Q = SGDFunctionApproximator(env)  # init Q function

        if h_model_to_load is not None:
            print(f'Loaded pretrained H model: {h_model_to_load}')
            self.load_model(filename=h_model_to_load)
        else:
            self.H = SGDFunctionApproximator(env)  # init Q function

        # init eligibility module
        out_dim = sum(
            t[1].n_components for t in self.Q.featurizer.transformer_list)
        self.trace_module = EligibilityModule(
            out_dim, env.action_space.n, trace_decay, trace_scaling, trace_accum, control_sharing)

        # hyperparameters
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.min_eps = min_eps

        # calculate episodic reduction in epsilon
        self.epsilon_step = (epsilon - min_eps) / num_episodes

        # tamer log path
        tamer_log_path = os.path.join(
            self.logs_dir, "tamer", f'{self.uuid}.csv')
        # episode log path
        episode_log_path = os.path.join(
            self.logs_dir, "episode", f'{self.uuid}.csv')

        # Logger
        self.logger = Logger(episode_log_path, tamer_log_path)

    def act(self, state):
        """ Epsilon-greedy Policy """
        n_actions = self.env.action_space.n
        feature_vector = self.Q.featurize_state(state)
        h_preds = np.array(self.H.predict(state))
        q_preds = np.array(self.Q.predict(state))

        # Exploration
        if np.random.random() < self.epsilon:
            return np.random.randint(n_actions)

        # Exploitation
        if self.control_sharing:
            beta = self.trace_module.compute_beta(feature_vector)
            if np.random.rand() < min(1, beta):
                return np.argmax(h_preds)
            else:
                return np.argmax(q_preds)

        else:
            # Action-biasing: compute beta per action
            betas = np.array([self.trace_module.compute_beta(feature_vector, a)
                              for a in range(n_actions)])
            biased_q_preds = q_preds + betas * h_preds
            return np.argmax(biased_q_preds)

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

            # Update Q
            if done and next_state[0] >= 0.5:
                td_target = reward
            else:
                td_target = reward + self.discount_factor * np.max(
                    self.Q.predict(next_state)
                )
            self.Q.update(state, action, td_target)

            # Collect human reward and update H
            human_reward = 0
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

                # update trace for state-action and decay for all traces except action
                self.trace_module.update_trace(
                    self.Q.featurize_state(state), action)
                self.trace_module.decay_trace(action)

            # if no feedback signal, then decay all traces
            if human_reward == 0:
                self.trace_module.decay_trace()

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
        disp = Interface(action_map=self.action_map,
                         env_frame_shape=self.env.render().shape)
        self.trace_module.reset_trace()
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
            print(f'\nSaving Q Model to q_{model_file_to_save}')
            self.save_model(filename=f'q_{model_file_to_save}', model=self.Q)
            print(f'\nSaving H Model to h_{model_file_to_save}')
            self.save_model(filename=f'h_{model_file_to_save}', model=self.H)

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
            disp = Interface(action_map=self.action_map,
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

    def save_model(self, filename, model):
        """
        Saves a model to models dir
        Args:
            filename: name of pickled file
        """
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(self.models_dir.joinpath(filename), 'wb') as f:
            pickle.dump(model, f)

    def load_model(self, filename):
        """
        Load a model from models dir
        Args:
            filename: name of pickled file
        """
        filename = filename + '.p' if not filename.endswith('.p') else filename
        with open(self.models_dir.joinpath(filename), 'rb') as f:
            model = pickle.load(f)

        return model
