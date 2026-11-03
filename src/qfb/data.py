"""Asset return data: a seeded synthetic sector-factor model, or your own CSV of prices."""
from __future__ import annotations

import numpy as np
import pandas as pd


def synthetic_returns(
    num_assets: int, num_days: int = 500, num_sectors: int = 2, seed: int = 0
) -> pd.DataFrame:
    """Daily returns from a one-market + sector factor model, so assets in the same sector
    are correlated. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    sector = np.arange(num_assets) % num_sectors
    market = rng.normal(0.0003, 0.010, num_days)
    sectors = rng.normal(0.0, 0.008, (num_days, num_sectors))
    beta = rng.uniform(0.6, 1.4, num_assets)
    alpha = rng.normal(0.0004, 0.0003, num_assets)
    idio = rng.normal(0.0, 0.006, (num_days, num_assets))
    r = alpha + market[:, None] * beta + sectors[:, sector] + idio
    cols = [f"A{i}_S{s}" for i, s in enumerate(sector)]
    return pd.DataFrame(r, columns=cols)


def returns_from_price_csv(path: str) -> pd.DataFrame:
    """Load a CSV of prices (rows = dates, columns = tickers) and return daily log returns."""
    prices = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    return np.log(prices).diff().dropna()


def annualized_stats(returns: pd.DataFrame, periods: int = 252) -> tuple[np.ndarray, np.ndarray]:
    """Expected returns mu and covariance Sigma, annualized."""
    return returns.mean().to_numpy() * periods, returns.cov().to_numpy() * periods
