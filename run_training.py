from src.data_collector import fetch_price_data, compute_returns, save_to_disk
from src.regime_detector import load_features, fit_hmm_and_label, save_outputs
from src.rl_agent import train_ppo
from src.backtester import evaluate_model
import pandas as pd

# 1. Collect data & save CSVs
prices = fetch_price_data()
returns, features = compute_returns(prices)
save_to_disk(prices, returns, features)

# 2. Detect regimes
feat_loaded = load_features()
hmm, labeled = fit_hmm_and_label(feat_loaded)
save_outputs(hmm, labeled)

# 3. Train PPO agent
train_ppo()

# 4. Backtest trained agent
evaluate_model()
