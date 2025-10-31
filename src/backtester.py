import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path
from .portfolio_env import PortfolioEnv
from .rl_agent import make_env

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"

def evaluate_model():
    # load trained model
    model = PPO.load(DATA_DIR / "ppo_portfolio.zip")

    # build test env (last ~250 days only)
    test_env_fn = make_env(slice(-250, None))
    test_env = test_env_fn()

    obs, _ = test_env.reset()
    equity_curve = [1.0]

    weights_history = []

    for _ in range(200):  # ~200 days
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = test_env.step(action)

        new_value = equity_curve[-1] * (1.0 + info["raw_return"] - 0.0005 * info["turnover"])
        equity_curve.append(new_value)
        weights_history.append(info["weights"])

        if term or trunc:
            break

    equity_curve = np.array(equity_curve)

    # compute basic metrics
    total_return = equity_curve[-1] - equity_curve[0]
    daily_returns = equity_curve[1:] / equity_curve[:-1] - 1.0
    sharpe = (np.mean(daily_returns) / (np.std(daily_returns) + 1e-9)) * np.sqrt(252)
    max_drawdown = np.max(np.maximum.accumulate(equity_curve) - equity_curve) / np.max(np.maximum.accumulate(equity_curve))

    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame({
        "equity_curve": equity_curve
    }).to_csv(REPORT_DIR / "equity_curve.csv", index=False)

    print("Total Return:", total_return)
    print("Sharpe:", sharpe)
    print("Max Drawdown:", max_drawdown)

if __name__ == "__main__":
    evaluate_model()
