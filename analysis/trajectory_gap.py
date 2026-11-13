import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pystk2

from src import config
from src.env import make_env, vector_to_action
from src.evaluate import save_rows
from src.expert import Expert
from src.policy import load_learner
from src.sac import load_actor

PARACHUTE = pystk2.Attachment.Type.PARACHUTE.value
COLORS = {"Expert": "#5f5f5a", "BC": "#eb6834", "DAgger p=0.5": "#2a78d6", "RLPD": "#1baf7a", "DAgger p=0": "#eda100", "SAC only": "#e87ba4"}


def record(env, policy, name):
    obs, _ = env.reset(seed=config.EVAL_SEED)
    policy.reset()
    rows, done, step = [], False, 0
    while not done:
        a = policy.act(obs)
        obs, r, terminated, truncated, info = env.step(vector_to_action(a))
        done = terminated or truncated
        step += 1
        rows.append({"policy": name, "step": step, "time": step * config.DT, "distance": info["distance"],
                     "steer": float(a[0]), "throttle": float(a[1]), "parachute": int(obs["attachment"] == PARACHUTE)})
    return rows


def column(rows, key):
    return np.array([r[key] for r in rows])


def plot(runs, label, path):
    grid = np.linspace(0, min(column(r, "distance").max() for r in runs.values()), 1000)
    # t(x) : premier instant où le kart atteint la distance x (distance rendue croissante)
    time_at = {n: np.interp(grid, np.maximum.accumulate(column(r, "distance")), column(r, "time")) for n, r in runs.items()}

    fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
    for ax in axes:
        ax.grid(color="#e5e5e2", linewidth=0.6)
        ax.spines[["top", "right"]].set_visible(False)
        para = column(runs["Expert"], "parachute").astype(bool)
        dist = column(runs["Expert"], "distance")
        ax.axvspan(dist[para].min(), dist[para].max(), color="#d9d9d4", alpha=0.5, linewidth=0)

    axes[0].axhline(0, color="#8a8a85", linewidth=0.8)
    axes[0].plot(grid, time_at[label] - time_at["Expert"], color=COLORS[label], linewidth=1.5)
    axes[0].set_ylabel(f"Time gap\n{label} − expert (s)")
    axes[0].text(0.01, 0.06, f"< 0: {label} ahead", transform=axes[0].transAxes, color="#5f5f5a", fontsize=8)
    axes[0].text(dist[para].mean(), 1.02, "parachute", transform=axes[0].get_xaxis_transform(), ha="center", va="bottom", color="#5f5f5a", fontsize=8)

    for ax, key, label in [(axes[1], "steer", "Steering"), (axes[2], "throttle", "Throttle")]:
        for name, rows in runs.items():
            ax.plot(column(rows, "distance"), column(rows, key), color=COLORS[name], linewidth=1.0, label=name)
        ax.set_ylabel(label)
    axes[2].set_xlabel("Distance travelled (m)")
    fig.legend(*axes[1].get_legend_handles_labels(), frameon=False, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path)


def main(model, label, csv_path, fig_path):
    env = make_env()
    runs = {"Expert": record(env, Expert(), "Expert"), label: record(env, (load_actor if model.endswith("actor.pt") else load_learner)(model), label)}
    env.close()
    save_rows(csv_path, runs["Expert"] + runs[label])
    plot(runs, label, fig_path)
    for name, rows in runs.items():
        print(f"{name} | temps : {rows[-1]['time']:.1f} s | pas sous parachute : {sum(r['parachute'] for r in rows)}")
    print(f"Données : {csv_path} | figure : {fig_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/bc/seed_0/model.pt")
    parser.add_argument("--label", default="BC", choices=[k for k in COLORS if k != "Expert"])
    parser.add_argument("--csv", default="runs/analysis/trajectory_gap.csv")
    parser.add_argument("--fig", default="report/figures/trajectory_gap.pdf")
    args = parser.parse_args()
    main(args.model, args.label, args.csv, args.fig)
