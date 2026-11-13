import argparse

import pystk2

from src import config
from src.env import OBS_DIM, make_env, obs_to_vector, vector_to_action
from src.expert import Expert
from src.policy import load_learner

ATTACHMENTS = {v.value: k for k, v in pystk2.Attachment.Type.__members__.items()}
POWERUPS = {v.value: k for k, v in pystk2.Powerup.Type.__members__.items()}
BUTTONS = ("brake", "drift", "nitro", "fire", "rescue")




def main(render, model_path):
    env = make_env(render)
    policy = Expert() if model_path is None else load_learner(model_path)

    obs, _ = env.reset(seed=0)
    policy.reset()
    x = obs_to_vector(obs)
    assert x.shape == (OBS_DIM,), x.shape
    print(f"Vecteur d'observation : shape={x.shape}")

    done, steps, ret = False, 0, 0.0
    previous, buttons_sent = None, set()
    while not done:
        action = vector_to_action(policy.act(obs))
        buttons_sent |= {b for b in BUTTONS if action[b]}
        obs, r, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        steps += 1
        ret += r

        current = (ATTACHMENTS[obs["attachment"]], POWERUPS[obs["powerup_type"]])
        if current != previous:
            print(f"t={steps * config.DT:5.1f} s | pénalité attachée : {current[0]:<16} | objet en stock : {current[1]}")
            previous = current
    env.close()

    print(f"Boutons envoyés au moins une fois : {sorted(buttons_sent) or 'aucun'}")
    print(f"Tour fini : {terminated} | pas : {steps} | temps : {steps * config.DT:.1f} s "
          f"| distance : {info['distance']:.0f} m | retour : {ret:.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--model", default=None, help="ex. runs/bc/seed_0/model.pt (expert par défaut)")
    args = parser.parse_args()
    main(args.render, args.model)
