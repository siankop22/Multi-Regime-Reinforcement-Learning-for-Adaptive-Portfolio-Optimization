import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

REPORT_DIR = Path(__file__).resolve().parents[0]

df = pd.read_csv(REPORT_DIR / "equity_curve.csv")
plt.plot(df["equity_curve"])
plt.title("Equity Curve (RL Strategy)")
plt.xlabel("Day")
plt.ylabel("Portfolio Value")
plt.savefig(REPORT_DIR / "equity_curve.png", dpi=300)
print("Saved reports/equity_curve.png")
