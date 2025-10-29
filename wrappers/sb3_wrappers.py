import numpy as np
import gymnasium as gym
from gymnasium import spaces

class DecodedObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

    def observation(self, obs):
        x, y, b, p = self.env._decode(int(obs))
        x_n = float(x) / max(1.0, float(self.env.width - 1))
        y_n = float(y) / max(1.0, float(self.env.height - 1))
        b_n = float(b) / max(1.0, float(self.env.max_battery))
        p_f = float(p)
        return np.array([x_n, y_n, b_n, p_f], dtype=np.float32)

class OneHotObservationWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = int(env.nS)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(n,), dtype=np.float32)

    def observation(self, obs):
        n = int(self.env.nS)
        v = np.zeros((n,), dtype=np.float32)
        v[int(obs)] = 1.0
        return v
