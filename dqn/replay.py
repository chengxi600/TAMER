from collections import deque, namedtuple
import random


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
HumanTransition = namedtuple(
    'HumanTransition', ('state', 'action', 'feedback'))


class ReplayBuffer:
    def __init__(self, capacity, transition_type):
        self.memory = deque([], maxlen=capacity)
        self.transition_type = transition_type

    def push(self, *args):
        """Save a transition"""
        self.memory.append(self.transition_type(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
