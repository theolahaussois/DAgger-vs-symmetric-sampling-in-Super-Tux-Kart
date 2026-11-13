from pathlib import Path

from omegaconf import OmegaConf

from src.expert.kart_agent import KartAgent
from src.expert.agent_center import AgentCenter
from src.expert.agent_speed import AgentSpeed
from src.expert.agent_rescue import AgentRescue
from src.expert.agent_virage import AgentVirage


class Agent1(KartAgent):
    def __init__(self, env=None, path_lookahead=3):
        super().__init__(env)
        self.path_lookahead = path_lookahead
        self.conf = OmegaConf.load(Path(__file__).parent / "ConfigFileTeam1.yaml")

        self.agentCenter = AgentCenter(env, self.conf, path_lookahead)
        self.agentSpeed = AgentSpeed(env, self.conf, self.agentCenter, path_lookahead)
        # Nous avons retiré le drift et l'utilisation d'objets pour avoir les mêmes actions possibles que l'élève
        self.agentRescue = AgentRescue(env, self.conf, self.agentSpeed)
        self.agentVirage = AgentVirage(env, self.conf, self.agentRescue)

    def choose_action(self, obs):
        action = self.agentRescue.choose_action(obs)
        action = self.agentVirage.gestion_virage(obs, action)
        return action
