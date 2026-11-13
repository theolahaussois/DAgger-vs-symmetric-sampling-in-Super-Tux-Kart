import numpy as np

from src.expert.agent1 import Agent1


def _scalar(x):
    return float(np.asarray(x, dtype=np.float32).reshape(-1)[0])


class Expert:
    def reset(self):
        self.agent = Agent1()  # pour remettre à zéro ses états internes

    def act(self, obs):
        a = self.agent.choose_action(obs)
        steer = np.clip(_scalar(a["steer"]), -1.0, 1.0)
        # On projette les actions sur l'espace des actions de l'élève
        throttle = -1.0 if bool(a["brake"]) else np.clip(_scalar(a["acceleration"]), 0.0, 1.0)
        return np.array([steer, throttle], dtype=np.float32)
