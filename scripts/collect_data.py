import argparse
from pathlib import Path

import numpy as np

from src import config
from src.env import make_env, obs_to_vector, vector_to_action
from src.expert import Expert


def main(episodes, out):
    env = make_env()
    expert = Expert()
    keys = ["obs", "action", "reward", "next_obs", "terminated", "episode"]
    data = {k: [] for k in keys}

    for ep in range(episodes):
        obs, _ = env.reset(seed=config.COLLECT_SEED + ep)
        expert.reset()
        done, steps, ret = False, 0, 0.0
        while not done:
            a = expert.act(obs)
            next_obs, r, terminated, truncated, info = env.step(vector_to_action(a))

            data["obs"].append(obs_to_vector(obs))
            data["action"].append(a)
            data["reward"].append(r)
            data["next_obs"].append(obs_to_vector(next_obs))
            data["terminated"].append(terminated)  # pas truncated : la coupure de temps n'est pas un état terminal
            data["episode"].append(ep)

            obs = next_obs
            done = terminated or truncated
            steps += 1
            ret += r
        print(f"Épisode {ep + 1}/{episodes} | tour fini : {terminated} | temps : {steps * config.DT:.1f} s "
              f"| distance : {info['distance']:.0f} m | retour : {ret:.1f}")
    env.close()

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **{k: np.array(v, dtype=np.float32) for k, v in data.items()})
    print(f"{len(data['obs'])} transitions sauvegardées dans {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--out", default=config.EXPERT_DATASET)
    args = parser.parse_args()
    main(args.episodes, args.out)
