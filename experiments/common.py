from pathlib import Path

RESULTS = Path(__file__).resolve().parents[1] / "results"
RESULTS.mkdir(exist_ok=True)

# Default option: at-the-money European call, 1 year.
OPTION = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)
