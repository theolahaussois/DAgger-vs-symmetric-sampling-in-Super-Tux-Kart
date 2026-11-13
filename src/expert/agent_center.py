from src.expert.kart_agent import KartAgent
import numpy as np

class AgentCenter(KartAgent):
    def __init__(self, env, conf, path_lookahead=3):
        super().__init__(env)
        self.conf = conf
        self.dist = self.conf.dist
        self.ajust = self.conf.ajust
        self.path_lookahead = path_lookahead
        self.path_decalage = -1

    def observation_next_item(self, obs, action) :
        if obs["distance_down_track"] < 1 :
            nb_noeuds = len(obs["paths_end"])
            if (nb_noeuds == 83) and len(obs["items_position"]) > 0 :  # ajout : liste d'objets vide
                next_item = obs["items_position"][0]
                if abs(next_item[self.conf.x]) < 10 :
                    nb_noeuds = 1

            if (nb_noeuds in self.conf.maps_zero) :
                self.path_decalage = -3
            elif (nb_noeuds in self.conf.maps_one) :
                self.path_decalage = -2
            elif (nb_noeuds in self.conf.maps_three) :
                self.path_decalage = 0

        for i in range(len(obs["items_type"])) :
            vecteur_item = obs["items_position"][i]
            type_item = obs["items_type"][i]
            if (2 < vecteur_item[self.conf.z] < 15) and (abs(vecteur_item[self.conf.y]) < 1) :
                if (abs(vecteur_item[self.conf.x]) < 1.5) and type_item in self.conf.obstacles :
                    action = self.dodge_obstacle(obs, action, i)
                    return action
        return action

    def dodge_obstacle(self, obs, action, index) :
        vecteur_item = obs["items_position"][index]
        next_node = obs["paths_end"][self.path_lookahead]
        sign_next_node = -1 if next_node[self.conf.x] < 0 else 1
        if abs(next_node[self.conf.x]) > 10 :
            action["steer"] += 1 * sign_next_node
        else :
            if vecteur_item[self.conf.x] >= 0 :
                action["steer" ] -= 1
            else :
                action["steer"] += 1
        action["steer"] = np.clip(action["steer"], -1, 1)
        return action

    def take_bonus(self, obs, action, index) :
        vecteur_item = obs["items_position"][index]
        next_node = obs["paths_end"][self.path_lookahead]
        if vecteur_item[self.conf.x] * next_node[self.conf.x] < 0 :
            return action

        steer = vecteur_item[self.conf.x]
        steer = np.clip(steer, -1, 1)
        action["steer"] = steer
        return action

    def evite_ennemi(self, obs, action) :
        for kart in obs["karts_position"] :
            if abs(kart[self.conf.x]) < 1 and 0 < kart[self.conf.z] < 3:
                if action["steer"] >= 0 :
                    action["steer"] += 0.3
                else :
                    action["steer"] -= 0.3
                action["steer"] = np.clip(action["steer"], -1, 1)
                return action
        return action

    def path_ajust(self, obs, action):
        steer = action["steer"]
        center = obs["paths_end"][self.path_lookahead + self.path_decalage]
        if (center[self.conf.z] > 20 and abs(obs["center_path_distance"]) < 3) :
            steer = 0
        elif abs(center[self.conf.x]) > self.dist :
            steer += self.ajust * center[0]
        action["steer"] = np.clip(steer, -1, 1)
        return action

    def choose_action(self, obs):
        action = {
            "acceleration": 0,
            "steer": 0,
            "brake": False,
            "drift": False,
            "nitro": False,
            "rescue": False,
            "fire": False,
        }
        action = self.path_ajust(obs, action)
        action = self.evite_ennemi(obs, action)
        action = self.observation_next_item(obs, action)
        return action
