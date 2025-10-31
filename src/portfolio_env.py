import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

WINDOW = 5           # how many past days we feed as state
TRANSACTION_COST = 0.0005  # 5 bps per rebalance

class PortfolioEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, returns_df: pd.DataFrame, regimes: np.ndarray):
        super().__init__()

        self.returns_df = returns_df
        self.regimes = regimes
        self.assets = returns_df.columns.tolist()
        self.n_assets = len(self.assets)

        # action: continuous weights for each asset
        # we'll let the model output unconstrained R^n and we'll softmax it
        self.action_space = spaces.Box(
            low=-5.0,
            high=5.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        # observation: last WINDOW days of returns + current regime (1 value)
        obs_dim = self.n_assets * WINDOW + 1
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        self.t = None
        self.prev_w = None  # last allocation weights

    def _get_obs(self):
        # stack last WINDOW days of returns
        past_window = self.returns_df.iloc[self.t-WINDOW+1:self.t+1].values  # shape (WINDOW, n_assets)
        flat = past_window.flatten()  # length WINDOW * n_assets
        reg = np.array([self.regimes[self.t]], dtype=np.float32)
        obs = np.concatenate([flat, reg]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # start somewhere after WINDOW
        self.t = WINDOW
        self.prev_w = np.ones(self.n_assets) / self.n_assets
        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        exp_a = np.exp(action - np.max(action))
        w = exp_a / np.sum(exp_a)

        r_t = self.returns_df.iloc[self.t].values
        gross_ret = np.dot(w, r_t)

        turnover = np.sum(np.abs(w - self.prev_w))
        cost = TRANSACTION_COST * turnover
        reward = gross_ret - cost

        self.prev_w = w
        self.t += 1
        terminated = self.t >= len(self.returns_df) - 1
        truncated = False

        if terminated:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        else:
            obs = self._get_obs()

        info = {
            "weights": w,
            "raw_return": gross_ret,
            "turnover": turnover
        }

        return obs, reward, terminated, truncated, info


    def render(self):
        pass  # optional
