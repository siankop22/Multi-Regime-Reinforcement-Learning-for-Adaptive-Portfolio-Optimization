import pandas as pd
import numpy as np
from hmmlearn.hmm import GaussianHMM
from pathlib import Path
import joblib  # for saving model

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
MODEL_DIR = Path(__file__).resolve().parents[1] / "data"

def load_features():
    feat = pd.read_csv(DATA_DIR / "features.csv", index_col=0, parse_dates=True)
    return feat

def fit_hmm_and_label(feat: pd.DataFrame, n_components=3):
    # we'll use two core drivers: mean return across assets, and rolling avg corr
    asset_cols = [c for c in feat.columns if "_" not in c and c != "avg_corr"]
    mean_ret = feat[asset_cols].mean(axis=1)
    X = np.column_stack([
        mean_ret.values,
        feat["avg_corr"].values
    ])

    hmm = GaussianHMM(
        n_components=n_components,
        covariance_type="full",
        n_iter=500,
        random_state=42
    )
    hmm.fit(X)
    hidden_states = hmm.predict(X)

    out = feat.copy()
    out["regime"] = hidden_states  # 0/1/2

    return hmm, out

def save_outputs(hmm, labeled_feat: pd.DataFrame):
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(hmm, MODEL_DIR / "hmm_model.pkl")
    labeled_feat.to_csv(MODEL_DIR / "features_with_regime.csv")

if __name__ == "__main__":
    feat = load_features()
    hmm, labeled = fit_hmm_and_label(feat)
    save_outputs(hmm, labeled)
    print("Saved HMM model and features_with_regime.csv")
