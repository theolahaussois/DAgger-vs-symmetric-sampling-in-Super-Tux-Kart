from src.env import make_env
from src.evaluate import evaluate, save_rows
from src.expert import Expert


def main():
    env = make_env()
    row = evaluate(env, Expert())
    env.close()
    save_rows("runs/expert/eval.csv", [row])
    print(f"Expert | fini : {row['finished']} | temps : {row['lap_time']:.1f} s | "
          f"distance : {row['distance']:.0f} m | retour : {row['return']:.2f}")


if __name__ == "__main__":
    main()
