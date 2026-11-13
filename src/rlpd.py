import numpy as np
import torch

from src.env import ACT_DIM, OBS_DIM

KEYS = {"obs": OBS_DIM, "action": ACT_DIM, "reward": None, "next_obs": OBS_DIM, "terminated": None}


class Buffer:
    def __init__(self, capacity):
        self.data = {k: np.zeros((capacity, d) if d else capacity, dtype=np.float32) for k, d in KEYS.items()}
        self.size = 0

    @classmethod
    def from_dataset(cls, dataset):
        buffer = cls(len(dataset["obs"]))
        for k in KEYS:
            buffer.data[k][:] = dataset[k]
        buffer.size = len(dataset["obs"])
        return buffer

    def add(self, **transition):
        for k, v in transition.items():
            self.data[k][self.size] = v
        self.size += 1

    def sample(self, n, rng):
        idx = rng.integers(self.size, size=n)
        return {k: v[idx] for k, v in self.data.items()}


def sample_batch(offline, online, batch_size, offline_ratio, rng):
    n_offline = int(batch_size * offline_ratio)  # moitié démonstrations, moitié en ligne
    parts = [online.sample(batch_size - n_offline, rng)]
    if n_offline > 0:
        parts.append(offline.sample(n_offline, rng))
    return {k: torch.as_tensor(np.concatenate([p[k] for p in parts])) for k in KEYS}
