from csv import DictWriter


class Logger:
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
            'Ep start ts',
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

    def log_tamer_step(self, episode_idx, ep_start_ts, feedback_ts, human_reward, env_reward):
        self.tamer_writer.writerow(
            {
                'Episode': episode_idx + 1,
                'Ep start ts': ep_start_ts,
                'Feedback ts': feedback_ts,
                'Human Reward': human_reward,
                'Environment Reward': env_reward,
            }
        )

    def log_episode(self, episode_idx, ep_start_ts, ep_end_ts, total_reward):
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
