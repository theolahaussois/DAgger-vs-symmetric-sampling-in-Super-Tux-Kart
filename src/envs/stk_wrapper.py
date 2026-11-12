import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pystk

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
        obs = np.array([
            kart.velocity.x, kart.velocity.y, kart.velocity.z,
            kart.location.x, kart.location.y, kart.location.z,
            kart.rotation.x, kart.rotation.y, kart.rotation.z, kart.rotation.w,
            kart.front.x, kart.front.y, kart.front.z,
            kart.distance_down_track
        ], dtype=np.float32)
        return obs

    def reset(self, seed, options):
        super().reset(seed=seed)
        self.current_step = 0
        self.race.start()
        return self._get_obs(self.race.state), {}

    def step(self, action: np.ndarray):
        self.current_step += 1        
        act = pystk.Action()
        act.steer = float(action[0])
        act.acceleration = float(action[1])
        act.brake = False
        act.drift = False        
        self.race.step(act)
        state = self.race.state
        kart = state.karts[0]        
        reward = float(kart.speed)        
        terminated = bool(kart.distance_down_track > 0.99)
        truncated = bool(self.current_step >= self.max_steps)        
        return self._get_obs(state), reward, terminated, truncated, {}

    def close(self):
        self.race.stop()
        pystk.clean()