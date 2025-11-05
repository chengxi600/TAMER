from csv import DictWriter
import requests
from typing import List
from pydantic import BaseModel
from datetime import datetime
from datetime import time
from collections import defaultdict


class TimeStepLog(BaseModel):
    """ 
    Represents a log entry for a single time step
    """
    time: int
    env_reward: float
    human_reward: float


class EpisodeLog(BaseModel):
    """
    Represents a log entry for a single episode
    """
    episode: int
    total_reward: float
    ep_start_ts: time
    ep_end_ts: time
    ts_logs: List[TimeStepLog]


class ExperimentLog(BaseModel):
    """
    Represents a log entry for an experiment
    """
    name: str
    date: str
    algorithm: str
    episode_logs: List[EpisodeLog]


def init_episode():
    return {
        "episode": None,
        "ts_logs": [],
        "total_reward": 0,
        "ep_start_ts": None,
        "ep_end_ts": None,
    }


class Logger:
    def __init__(self, episode_log_path=None, tamer_log_path=None, log_csv=False):
        """The logger allows for writing logs to a CSV file and saving logs to the MongoDB database.

        Args:
            episode_log_path (_type_, optional): file path for the episode logs. Defaults to None.
            tamer_log_path (_type_, optional): file path for the timestep logs. Defaults to None.
            log_csv (bool, optional): whether to write to CSV. Defaults to False.
        """

        # mapping from episode idx to an EpisodeLog
        self.episodes: dict[int, dict] = defaultdict(init_episode)
        self.log_csv = log_csv

        if log_csv:
            self.csv_logger = CSVLogger(episode_log_path, tamer_log_path)

    def log_tamer_step(self, episode_idx, feedback_ts, human_reward, env_reward):
        """ Logs a single timestep. Saves a TimeStepLog entry to the Logger. 
            If log_csv enabled, writes timestep logs to a csv file.

        Args:
            episode_idx (int): episode index
            feedback_ts (str): feedback timestamp
            human_reward (int): human reward
            env_reward (int): environment reward
        """
        timestep_log = TimeStepLog(
            time=feedback_ts,
            env_reward=env_reward,
            human_reward=human_reward
        )
        self.episodes[episode_idx]["ts_logs"].append(timestep_log)

        if self.log_csv:
            self.csv_logger.log_tamer_step_csv(
                episode_idx, feedback_ts, human_reward, env_reward)

    def log_episode(self, episode_idx, ep_start_ts, ep_end_ts, total_reward):
        """ Logs a single episode. Saves a EpisodeLog entry to the Logger. 
            If log_csv enabled, writes episodic logs to a csv file.

        Args:
            episode_idx (int): episode index
            ep_start_ts (str): episode start ts
            ep_end_ts (str): episode end ts
            total_reward (int): total episode environment reward 
        """
        self.episodes[episode_idx]["episode"] = episode_idx
        self.episodes[episode_idx]["ep_start_ts"] = ep_start_ts
        self.episodes[episode_idx]["ep_end_ts"] = ep_end_ts
        self.episodes[episode_idx]["total_reward"] = total_reward

        if self.log_csv:
            self.csv_logger.log_episode_csv(episode_idx, ep_start_ts,
                                            ep_end_ts, total_reward)

    def log_experiment(
        self,
        name,
        date,
        algorithm,
    ):
        """ Writes an ExperimentLog into the MongoDB database using collected EpisodeLogs and TimeStepLogs.

        Args:
            name (str): name of the current Experiment
            date (str): date of experiment
            algorithm (str): algorithm used for this experiment. 
        """
        url = 'https://hfrl-dashboard.vercel.app/api/log'

        episode_logs = [
            EpisodeLog(**ep_dict) for ep_dict in self.episodes.values()
        ]
        experiment_log = ExperimentLog(
            name=name,
            date=date,
            algorithm=algorithm,
            episode_logs=episode_logs
        )
        x = requests.post(url, json=experiment_log.model_dump(mode='json'))


class CSVLogger:
    def __init__(self, episode_log_path, tamer_log_path):
        self.episode_log_path = episode_log_path
        self.step_log_path = tamer_log_path

        self.episode_log_columns = [
            'Episode',
            'Ep start ts',
            'Ep end ts',
            'Avg eval reward',
        ]
        self.tamer_log_columns = [
            'Episode',
            'Feedback ts',
            'Human Reward',
            'Environment Reward',
        ]

        # open files
        self.episode_file = open(episode_log_path, 'a+', newline='')
        self.tamer_file = open(tamer_log_path, 'a+', newline='')

        # writers
        self.episode_writer = DictWriter(
            self.episode_file, fieldnames=self.episode_log_columns)
        self.tamer_writer = DictWriter(
            self.tamer_file, fieldnames=self.tamer_log_columns)

        self.episode_writer.writeheader()
        self.tamer_writer.writeheader()

    def log_tamer_step_csv(self, episode_idx, feedback_ts, human_reward, env_reward):
        self.tamer_writer.writerow(
            {
                'Episode': episode_idx + 1,
                'Feedback ts': feedback_ts,
                'Human Reward': human_reward,
                'Environment Reward': env_reward,
            }
        )

    def log_episode_csv(self, episode_idx, ep_start_ts, ep_end_ts, total_reward):
        self.episode_writer.writerow(
            {
                'Episode': episode_idx + 1,
                'Ep start ts': ep_start_ts,
                'Ep end ts': ep_end_ts,
                'Avg eval reward': total_reward,
            }
        )

    def close(self):
        self.episode_file.close()
        self.tamer_file.close()
