"""Classical reference: Black-Scholes price and a discretized log-normal model.

The quantum circuits can only load a finite grid of 2**n asset prices, so we price
against two references:
  * ``bs_call_price``: the continuous Black-Scholes closed form (model reference)
  * ``DiscreteModel.exact_price``: the exact expectation on the same grid the circuit
    loads (algorithm reference; QAE error is measured against this one)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import lognorm, norm


def bs_call_price(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """Black-Scholes price of a European call."""
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return float(S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))


@dataclass
class DiscreteModel:
    """Asset price at maturity, discretized onto 2**num_qubits grid points."""

    prices: np.ndarray        # grid of terminal prices S_T
    probs: np.ndarray         # probability of each grid point (sums to 1)
    payoff: np.ndarray        # call payoff max(S_T - K, 0) on the grid
    discount: float           # exp(-rT)

    @property
    def payoff_scale(self) -> float:
        """Max payoff, used to rescale payoffs into [0, 1] for amplitude encoding."""
        return float(self.payoff.max())

    @property
    def normalized_payoff(self) -> np.ndarray:
        return self.payoff / self.payoff_scale

    @property
    def exact_amplitude(self) -> float:
        """a = E[f(S_T)] with f = payoff / max payoff. This is what QAE estimates."""
        return float(np.dot(self.probs, self.normalized_payoff))

    def amplitude_to_price(self, a: float) -> float:
        return self.discount * self.payoff_scale * a

    @property
    def exact_price(self) -> float:
        return self.amplitude_to_price(self.exact_amplitude)


def lognormal_grid(
    S0: float, K: float, r: float, sigma: float, T: float, num_qubits: int, width: float = 5.0
) -> DiscreteModel:
    """Discretize the risk-neutral log-normal distribution of S_T onto 2**num_qubits points.

    The range mean +/- ``width`` standard deviations (truncated at 0) is split into 2**n equal
    bins; each grid point is a bin midpoint carrying that bin's exact probability mass (CDF
    difference), renormalized for the truncated tails. A wide range matters for a call because
    the payoff grows in the upper tail.
    """
    mu = np.log(S0) + (r - 0.5 * sigma**2) * T
    s = sigma * np.sqrt(T)
    dist = lognorm(s=s, scale=np.exp(mu))
    mean, std = dist.mean(), dist.std()
    low, high = max(1e-6, mean - width * std), mean + width * std
    edges = np.linspace(low, high, 2**num_qubits + 1)
    prices = 0.5 * (edges[:-1] + edges[1:])
    probs = np.diff(dist.cdf(edges))
    probs = probs / probs.sum()
    payoff = np.maximum(prices - K, 0.0)
    if payoff.max() <= 0:
        raise ValueError("Strike is above the whole grid; payoff is identically zero.")
    return DiscreteModel(prices=prices, probs=probs, payoff=payoff, discount=float(np.exp(-r * T)))
