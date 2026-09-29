from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class EpisodeBatch:
    observations: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    masks: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    next_states: np.ndarray
    next_masks: np.ndarray
    terminated: np.ndarray


class EpisodeReplayBuffer:
    def __init__(self, capacity: int, seed: int = 42):
        self.storage: deque[EpisodeBatch] = deque(maxlen=capacity)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, episode: EpisodeBatch) -> None:
        self.storage.append(episode)

    def sample(self, batch_size: int) -> list[EpisodeBatch]:
        if batch_size > len(self.storage):
            raise ValueError("batch_size exceeds stored episodes")
        indices = self.rng.choice(len(self.storage), size=batch_size, replace=False)
        return [self.storage[int(i)] for i in indices]

