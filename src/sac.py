import copy

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

from src import config
from src.env import ACT_DIM, OBS_DIM
from src.policy import Learner


def mlp(in_dim, out_dim, layer_norm):
    layers, dim = [], in_dim
    for _ in range(2):
        layers += [nn.Linear(dim, config.HIDDEN)] + ([nn.LayerNorm(config.HIDDEN)] if layer_norm else []) + [nn.ReLU()]
        dim = config.HIDDEN
    return nn.Sequential(*layers, nn.Linear(dim, out_dim))


class Normalized(nn.Module):
    def __init__(self, obs_mean, obs_std):
        super().__init__()
        self.register_buffer("mean", torch.as_tensor(obs_mean, dtype=torch.float32))
        self.register_buffer("std", torch.as_tensor(np.maximum(obs_std, 1e-2), dtype=torch.float32))

    def norm(self, x):
        return (x - self.mean) / self.std


class Actor(Normalized):
    def __init__(self, obs_mean, obs_std):
        super().__init__(obs_mean, obs_std)
        self.net = mlp(OBS_DIM, 2 * ACT_DIM, layer_norm=False)

    def dist(self, x):
        mu, log_std = self.net(self.norm(x)).chunk(2, dim=-1)
        return mu, log_std.clamp(-5, 2)

    def forward(self, x):
        return torch.tanh(self.dist(x)[0])  # action déterministe pour l'évaluation

    def sample(self, x):
        mu, log_std = self.dist(x)
        normal = Normal(mu, log_std.exp())
        u = normal.rsample()  # reparamétrisation
        a = torch.tanh(u)
        log_prob = (normal.log_prob(u) - torch.log(1 - a ** 2 + 1e-6)).sum(-1)  # changement de variable tanh
        return a, log_prob


class Critic(Normalized):
    def __init__(self, obs_mean, obs_std):
        super().__init__(obs_mean, obs_std)
        self.q1 = mlp(OBS_DIM + ACT_DIM, 1, layer_norm=True)  # LayerNorm borne l'extrapolation
        self.q2 = mlp(OBS_DIM + ACT_DIM, 1, layer_norm=True)

    def forward(self, x, a):
        h = torch.cat([self.norm(x), a], dim=-1)
        return self.q1(h).squeeze(-1), self.q2(h).squeeze(-1)


class SAC:
    def __init__(self, obs_mean, obs_std):
        self.actor = Actor(obs_mean, obs_std)
        self.critic = Critic(obs_mean, obs_std)
        self.target = copy.deepcopy(self.critic)
        self.log_alpha = torch.full((1,), float(np.log(config.INIT_ALPHA)), requires_grad=True)
        self.target_entropy = -ACT_DIM
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=config.SAC_LR)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=config.SAC_LR)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=config.SAC_LR)

    def act(self, x):
        with torch.no_grad():
            return self.actor.sample(torch.as_tensor(x)[None])[0][0].numpy()

    def update_critic(self, b):
        with torch.no_grad():
            a2, log_prob2 = self.actor.sample(b["next_obs"])
            q_target = torch.min(*self.target(b["next_obs"], a2)) - self.log_alpha.exp() * log_prob2
            y = b["reward"] + config.GAMMA * (1 - b["terminated"]) * q_target
        q1, q2 = self.critic(b["obs"], b["action"])
        loss = ((q1 - y) ** 2).mean() + ((q2 - y) ** 2).mean()
        self.critic_opt.zero_grad()
        loss.backward()
        self.critic_opt.step()
        with torch.no_grad():
            for p, p_target in zip(self.critic.parameters(), self.target.parameters()):
                p_target.mul_(1 - config.TAU).add_(config.TAU * p)
        return loss.item()

    def update_actor(self, obs):
        a, log_prob = self.actor.sample(obs)
        loss = (self.log_alpha.exp().detach() * log_prob - torch.min(*self.critic(obs, a))).mean()
        self.actor_opt.zero_grad()
        loss.backward()
        self.actor_opt.step()
        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()


def load_actor(path):
    actor = Actor(np.zeros(OBS_DIM), np.ones(OBS_DIM))
    actor.load_state_dict(torch.load(path))
    return Learner(actor)  # politique déterministe
