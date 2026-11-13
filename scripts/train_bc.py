import numpy as np
import torch

from src import config
from src.bc import train_bc
from src.env import make_env
from src.evaluate import evaluate, save_rows
from src.policy import MLP, Learner


def main():
    data = np.load(config.EXPERT_DATASET)
    obs, actions = data["obs"], data["action"]
    env = make_env()
    rows = []
    for seed in config.TRAIN_SEEDS:
        torch.manual_seed(seed)
        model = MLP(obs.mean(0), obs.std(0))
        losses = train_bc(model, obs, actions, config.BC_STEPS, config.BATCH_SIZE, config.LR)
        result = evaluate(env, Learner(model))

        run_dir = f"runs/bc/seed_{seed}"
        save_rows(f"{run_dir}/loss.csv", [{"step": s, "loss": l} for s, l in enumerate(losses) if s % config.LOG_EVERY == 0])
        torch.save(model.state_dict(), f"{run_dir}/model.pt")
        rows.append({"seed": seed, "final_loss": np.mean(losses[-config.LOG_EVERY:]), **result})
        print(f"graine {seed} | perte finale : {rows[-1]['final_loss']:.5f} | fini : {result['finished']} | "
              f"temps : {result['lap_time']:.1f} s | distance : {result['distance']:.0f} m | retour : {result['return']:.2f}")
    env.close()
    save_rows("runs/bc/eval.csv", rows)

    returns = np.array([r["return"] for r in rows])
    print(f"BC | tours finis : {sum(r['finished'] for r in rows)}/{len(rows)} | "
          f"retour : {returns.mean():.2f} ± {returns.std(ddof=1):.2f}")


if __name__ == "__main__":
    main()
