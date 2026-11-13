import numpy as np

from src import config
from src.env import obs_to_vector, vector_to_action


def rollout_with_labels(env, learner, expert, beta, rng):
    obs, _ = env.reset(seed=config.EVAL_SEED)
    learner.reset()
    expert.reset()
    states, labels = [], []
    done = False
    while not done:
        a_expert = expert.act(obs)
        states.append(obs_to_vector(obs))
        labels.append(a_expert)  # étiquette = action de l'expert, quel que soit le conducteur
        a = a_expert if rng.random() < beta else learner.act(obs)  # pi_i = beta pi* + (1 - beta) pi_hat
        obs, r, terminated, truncated, info = env.step(vector_to_action(a))
        done = terminated or truncated
    return np.array(states), np.array(labels)
