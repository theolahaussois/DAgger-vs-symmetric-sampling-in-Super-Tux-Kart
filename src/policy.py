import numpy as np
import torch
from torch import nn

from src import config
from src.env import ACT_DIM, OBS_DIM, obs_to_vector


class MLP(nn.Module):
    def __init__(self, obs_mean, obs_std):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(OBS_DIM, config.HIDDEN), nn.ReLU(),
            nn.Linear(config.HIDDEN, config.HIDDEN), nn.ReLU(),
            nn.Linear(config.HIDDEN, ACT_DIM), nn.Tanh(),
        )
        self.register_buffer("mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.maximum(obs_std, 1e-2), dtype=torch.float32))

    def forward(self, x):
        return self.net((x - self.mean) / self.std)


class Learner:
    def __init__(self, model):
        self.model = model

    def reset(self):
        pass

    def act(self, obs):
        with torch.no_grad():
            return self.model(torch.as_tensor(obs_to_vector(obs))).numpy()


def load_learner(path):
    model = MLP(np.zeros(OBS_DIM), np.ones(OBS_DIM))
    model.load_state_dict(torch.load(path))  # inclut la normalisation sauvegardée
    return Learner(model)
