import gymnasium as gym
import numpy as np
from pystk2_gymnasium import AgentSpec

from src import config

OBS_DIM = 8 + 4 * config.N_PATHS
ACT_DIM = 2


def make_env(render=False):
    return gym.make(
        "supertuxkart/full-v0",
        render_mode="human" if render else None,
        track=config.TRACK,
        num_kart=1,
        laps=1,
        agent=AgentSpec(use_ai=False, rank_start=0),
        max_episode_steps=config.MAX_STEPS,
    )

def obs_to_vector(obs):
    k = config.N_PATHS
    paths_end = np.array(obs["paths_end"][:k], dtype=np.float32).reshape(-1, 3)
    paths_width = np.array(obs["paths_width"][:k], dtype=np.float32).reshape(-1)
    return np.concatenate([
        obs["velocity"],
        obs["center_path"],
        obs["center_path_distance"],
        obs["max_steer_angle"],
        paths_end.reshape(-1),
        paths_width,
    ]).astype(np.float32)


def vector_to_action(a):
    steer, throttle = float(a[0]), float(a[1])
    return {
        "steer": np.array([steer], dtype=np.float32),
        "acceleration": np.array([max(throttle, 0.0)], dtype=np.float32),
        "brake": int(throttle < 0.0),  # throttle < 0 : frein / marche arrière
        "drift": 0,
        "nitro": 0,
        "fire": 0,
        "rescue": 0,
    }
