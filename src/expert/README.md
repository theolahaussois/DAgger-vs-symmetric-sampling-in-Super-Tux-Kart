# Expert

Code de l'agent `Agent1` (équipe 1, "Tasty Crousteam") copié depuis
[osigaud/pystk_project](https://github.com/osigaud/pystk_project/tree/master/src/agents),
avec ses hyper-paramètres d'origine (`ConfigFileTeam1.yaml`).

Modifications :
1. imports adaptés (`agents.*` -> `src.expert.*`) ;
2. `AgentItems` (utilisation des objets, nitro permanent) et `AgentDrift` (non appelé dans
   l'original) retirés, car l'étude porte sur un kart seul sans objets ;
3. `AgentCenter` : protection contre une liste d'objets vide ;
4. `Expert` (`__init__.py`) projette l'action sur l'espace de l'élève `(steer, throttle)` :
   le drift choisi par `AgentVirage` est ignoré, `brake` devient `throttle = -1`.

L'expert *exécuté* est donc exactement celui que l'élève peut imiter.
