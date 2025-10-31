import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path

ASSETS = ["SPY", "QQQ", "IWM", "TLT", "GLD"]  # equities, small-cap, bonds, gold
START_DATE = "2012-01-01"
END_DATE = "2024-01-01"

DATA_DIR = Path(__file__).resolve().parents[1] / "data"

def fetch_price_data():
    # force auto_adjust=False to keep column structure
    data = yf.download(ASSETS, start=START_DATE, end=END_DATE, auto_adjust=False)
    
    # if "Adj Close" exists, select it; otherwise use Close
    if ("Adj Close" in data.columns.get_level_values(0)):
        data = data["Adj Close"]
    else:
        data = data["Close"]
    
    data = data.dropna()
    return data


def compute_returns(prices: pd.DataFrame, window_vol=10):
    rets = prices.pct_change().dropna()
    vol = rets.rolling(window_vol).std()

    corr_list = []
    for i in range(len(rets)):
        if i < window_vol:
            corr_list.append(np.nan)
        else:
            c = rets.iloc[i-window_vol+1:i+1].corr().values
            tri_vals = c[np.triu_indices_from(c, k=1)]
            corr_list.append(np.nanmean(tri_vals))

    corr = pd.Series(corr_list, index=rets.index, name="avg_corr")

    feat = pd.concat([rets, vol.add_suffix("_vol"), corr], axis=1)
    feat = feat.replace([np.inf, -np.inf], np.nan).dropna()

    return rets.loc[feat.index], feat


def save_to_disk(prices, returns, features):
    DATA_DIR.mkdir(exist_ok=True)
    prices.to_csv(DATA_DIR / "prices.csv")
    returns.to_csv(DATA_DIR / "returns.csv")
    features.to_csv(DATA_DIR / "features.csv")

if __name__ == "__main__":
    prices = fetch_price_data()
    returns, features = compute_returns(prices)
    save_to_disk(prices, returns, features)
    print("Data saved to data/prices.csv, data/returns.csv, data/features.csv")
