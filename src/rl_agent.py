import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path
from .portfolio_env import PortfolioEnv, WINDOW

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "data"

def make_env(train_slice=slice(None, None)):
    rets = pd.read_csv(DATA_DIR / "returns.csv", index_col=0, parse_dates=True)
    labeled = pd.read_csv(DATA_DIR / "features_with_regime.csv", index_col=0, parse_dates=True)

    # align
    rets = rets.loc[labeled.index]

    rets_train = rets.iloc[train_slice]
    regimes_train = labeled["regime"].iloc[train_slice].values

    def _init():
        return PortfolioEnv(rets_train, regimes_train)
    return _init

def train_ppo():
    env = DummyVecEnv([make_env(slice(0, -250))])  # train on all but last ~250 days
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=200_000)

    model.save(MODEL_DIR / "ppo_portfolio.zip")
    print("Saved model to data/ppo_portfolio.zip")

if __name__ == "__main__":
    train_ppo()
