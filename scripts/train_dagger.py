import time

import numpy as np
import torch

from src import config
from src.bc import train_bc
from src.dagger import rollout_with_labels
from src.env import make_env
from src.evaluate import evaluate, save_rows
from src.expert import Expert
from src.policy import MLP, Learner


def run_seed(env, seed, expert_obs, expert_actions):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    expert = Expert()
    obs, actions = expert_obs, expert_actions
    curve, env_steps = [], 0
    for it in range(config.DAGGER_ITERS + 1):
        start = time.perf_counter()
        model = MLP(expert_obs.mean(0), expert_obs.std(0))  # réentraînement complet sur D (Follow-the-Leader)
        losses = train_bc(model, obs, actions, config.BC_STEPS, config.BATCH_SIZE, config.LR)
        result = evaluate(env, Learner(model))
        curve.append({"iteration": it, "env_steps": env_steps, "expert_labels": len(obs),
                      "final_loss": np.mean(losses[-config.LOG_EVERY:]), **result,
                      "wall_time": time.perf_counter() - start})
        print(f"graine {seed} | itération {it:2d} | données : {len(obs):5d} | fini : {result['finished']} | "
              f"temps : {result['lap_time']:.1f} s | retour : {result['return']:.2f}")
        if it == config.DAGGER_ITERS:
            break
        beta = config.DAGGER_P ** (it + 1)
        new_obs, new_actions = rollout_with_labels(env, Learner(model), expert, beta, rng)
        obs = np.concatenate([obs, new_obs])
        actions = np.concatenate([actions, new_actions])
        env_steps += len(new_obs)
    return model, curve


def main():
    data = np.load(config.EXPERT_DATASET)
    env = make_env()
    out = f"runs/dagger_p{config.DAGGER_P}"
    final = []
    for seed in config.TRAIN_SEEDS:
        model, curve = run_seed(env, seed, data["obs"], data["action"])
        save_rows(f"{out}/seed_{seed}/curve.csv", curve)
        torch.save(model.state_dict(), f"{out}/seed_{seed}/model.pt")
        final.append({"seed": seed, **curve[-1]})
    env.close()
    save_rows(f"{out}/eval.csv", final)

    returns = np.array([r["return"] for r in final])
    print(f"DAgger (p = {config.DAGGER_P}) | tours finis : {sum(r['finished'] for r in final)}/{len(final)} | "
          f"retour : {returns.mean():.2f} ± {returns.std(ddof=1):.2f}")


if __name__ == "__main__":
    main()
