import time

import numpy as np
import torch

from src import config
from src.env import make_env, obs_to_vector, vector_to_action
from src.evaluate import evaluate, save_rows
from src.policy import Learner
from src.rlpd import Buffer, sample_batch
from src.sac import SAC


def run_seed(env, seed, dataset):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    agent = SAC(dataset["obs"].mean(0), dataset["obs"].std(0))
    offline, online = Buffer.from_dataset(dataset), Buffer(config.RL_STEPS)
    curve, episodes, next_eval, start = [], 0, 0, time.perf_counter()

    def log_eval(t):
        result = evaluate(env, Learner(agent.actor))
        curve.append({"env_steps": t, "train_episodes": episodes, **result, "wall_time": time.perf_counter() - start})
        print(f"graine {seed} | pas : {t:6d} | épisodes : {episodes:3d} | fini : {result['finished']} | "
              f"temps : {result['lap_time']:.1f} s | retour : {result['return']:.2f} | {curve[-1]['wall_time']:.0f} s")

    obs, _ = env.reset(seed=10_000 * (seed + 1))
    for t in range(config.RL_STEPS):
        x = obs_to_vector(obs)
        a = agent.act(x)
        obs, r, terminated, truncated, _ = env.step(vector_to_action(a))
        online.add(obs=x, action=a, reward=r, next_obs=obs_to_vector(obs), terminated=terminated)

        if online.size >= config.BATCH_SIZE:
            for _ in range(config.UTD):
                batch = sample_batch(offline, online, config.BATCH_SIZE, config.OFFLINE_RATIO, rng)
                agent.update_critic(batch)
            agent.update_actor(batch["obs"])

        if terminated or truncated:
            episodes += 1
            if t + 1 >= next_eval:  # on évalue entre deux épisodes
                log_eval(t + 1)
                next_eval += config.EVAL_EVERY
            obs, _ = env.reset(seed=10_000 * (seed + 1) + episodes)
    log_eval(config.RL_STEPS)
    return agent, curve


def main():
    dataset = dict(np.load(config.EXPERT_DATASET))
    env = make_env()
    out = f"runs/rlpd_ratio{config.OFFLINE_RATIO}"
    final = []
    for seed in config.TRAIN_SEEDS:
        agent, curve = run_seed(env, seed, dataset)
        save_rows(f"{out}/seed_{seed}/curve.csv", curve)
        torch.save(agent.actor.state_dict(), f"{out}/seed_{seed}/actor.pt")
        final.append({"seed": seed, **curve[-1]})
    env.close()
    save_rows(f"{out}/eval.csv", final)

    returns = np.array([r["return"] for r in final])
    print(f"RLPD (part hors-ligne = {config.OFFLINE_RATIO}) | tours finis : {sum(r['finished'] for r in final)}/{len(final)} | "
          f"retour : {returns.mean():.2f} ± {returns.std(ddof=1):.2f}")


if __name__ == "__main__":
    main()
