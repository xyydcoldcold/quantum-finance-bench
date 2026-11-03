"""Mean-variance portfolio selection as a QUBO.

Choose exactly B of n assets (x_i = 1 means asset i is held, equal weights):

    min_x   q * x^T Sigma x  -  mu^T x  +  lam * (sum_i x_i - B)^2

q is the risk-aversion parameter and lam the penalty enforcing the budget constraint.
"""
from __future__ import annotations

import numpy as np

from .qubo import QUBO


def portfolio_qubo(mu: np.ndarray, sigma: np.ndarray, budget: int, risk_aversion: float = 0.5,
                   penalty: float | None = None) -> QUBO:
    n = len(mu)
    if penalty is None:  # pass penalty=0 when the constraint is enforced by the mixer instead
        # Large enough that breaking the budget is never worth it.
        penalty = float(np.abs(mu).sum() + risk_aversion * np.abs(sigma).sum()) + 1.0
    # (sum x - B)^2 = x^T 1 1^T x - 2B 1^T x + B^2
    Q = risk_aversion * sigma + penalty * np.ones((n, n))
    c = -mu - 2 * penalty * budget * np.ones(n)
    return QUBO(Q=Q, c=c, const=penalty * budget**2)


def is_feasible(x: np.ndarray, budget: int) -> np.ndarray:
    return np.atleast_2d(x).sum(axis=1) == budget
