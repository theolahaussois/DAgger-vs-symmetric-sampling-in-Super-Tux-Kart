from src.expert.kart_agent import KartAgent
import numpy as np

class AgentSpeed(KartAgent):
    def __init__(self, env, conf, agent, path_lookahead=3):
        super().__init__(env)
        self.conf = conf
        self.agent = agent
        self.path_lookahead = path_lookahead
        self.step_count = 0

    @staticmethod
    def detecter_virage(conf, obs):
        virage_serre = False
        nbsegments = min(conf.path_lookahead, len(obs["paths_start"]))
        for i in range(nbsegments):
            direction_segment = obs["paths_end"][i] - obs["paths_start"][i]
            diff_direction = direction_segment - obs["front"]
            ecart_direction = float(np.linalg.norm(diff_direction))
            distance_segment = abs(obs["paths_distance"][i][conf.x] - obs["paths_distance"][0][conf.x])

            if ecart_direction >= conf.ecartgrand and distance_segment < conf.dist_segment:
                virage_serre = True

        return virage_serre

    def ajuster_acceleration(self, virage_serre, act, obs):
        act["acceleration"] = max(act["acceleration"], 1)
        max_steer_angle = obs["max_steer_angle"]

        if not virage_serre:
            act["acceleration"] = self.conf.accel_ligne_droite

            direction_segment = obs["paths_end"][0] - obs["paths_start"][0]

        if max_steer_angle <= self.conf.max_steer_angle_petit:
            acceleration_freinee = act["acceleration"] - self.conf.frein_virage
            act["acceleration"] = np.clip(acceleration_freinee, 0.1, 1)

        elif max_steer_angle >= self.conf.max_steer_angle_grand:
            acceleration_boostee = act["acceleration"] + self.conf.accel_virage
            act["acceleration"] = np.clip(acceleration_boostee, 0.1, 1)

        direction_segment = obs["paths_end"][0] - obs["paths_start"][0]
        if direction_segment[1] > 0.05:
            act["acceleration"] = min(act["acceleration"] + 0.3, 1)
        elif direction_segment[1] < -0.05:
        	act["acceleration"] = max(act["acceleration"] - 0.1, 0.5)

        return act

    def choose_action(self, obs):
        act = self.agent.choose_action(obs)

        if self.step_count < 30:
            act["acceleration"] = 1
            act["brake"] = False
            self.step_count += 1
            return act

        virage_serre = self.detecter_virage(self.conf, obs)
        action_ajustee = self.ajuster_acceleration(virage_serre, act, obs)
        return action_ajustee
