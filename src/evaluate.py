import csv
from pathlib import Path

from src import config
from src.env import vector_to_action


def summarize(terminated, steps, info, ret):
    return {
        "finished": int(terminated),
        "steps": steps,
        "lap_time": steps * config.DT if terminated else float("nan"),
        "distance": info["distance"],
        "return": ret,
    }


def evaluate(env, policy):
    obs, _ = env.reset(seed=config.EVAL_SEED)
    policy.reset()
    done, steps, ret = False, 0, 0.0
    while not done:
        obs, r, terminated, truncated, info = env.step(vector_to_action(policy.act(obs)))
        done = terminated or truncated
        steps += 1
        ret += r
    return summarize(terminated, steps, info, ret)


def save_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
