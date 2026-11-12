import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pystk2 as pystk

class STKEnv(gym.Env):
    def __init__(self, track_name, max_steps):
        super().__init__()
        self.track_name = track_name
        self.max_steps = max_steps
        self.current_step = 0        
        config = pystk.GraphicsConfig.ld()
        config.screen_width = 128
        config.screen_height = 128
        pystk.init(config)       
        self.race_config = pystk.RaceConfig(num_kart=1, track=self.track_name)
        self.race = pystk.Race(self.race_config)        
        self.action_space = spaces.Box(
            low=np.array([-1.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0], dtype=np.float32)
        )        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

    def _get_obs(self, state) -> np.ndarray:
        kart = state.karts[0]        
        vel = kart.velocity
        loc = kart.location
        rot = kart.rotation
        front = kart.front
        dist = kart.distance_down_track        
        obs = np.array([
            vel[0], vel[1], vel[2],
            loc[0], loc[1], loc[2],
            rot[0], rot[1], rot[2], rot[3],
            front[0], front[1], front[2],
            dist
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.race.start()
        
        state = pystk.WorldState()
        state.update()
        
        return self._get_obs(state), {}

    def step(self, action: np.ndarray):
        self.current_step += 1        
        act = pystk.Action()
        act.steer = float(action[0])
        act.acceleration = float(action[1])
        act.brake = False
        act.drift = False        
        self.race.step(act)        
        state = pystk.WorldState()
        state.update()        
        kart = state.karts[0]        
        reward = float(kart.speed)
        terminated = bool(kart.distance_down_track > 0.99)
        truncated = bool(self.current_step >= self.max_steps)        
        return self._get_obs(state), reward, terminated, truncated, {}

    def close(self):
        self.race.stop()
        pystk.clean()