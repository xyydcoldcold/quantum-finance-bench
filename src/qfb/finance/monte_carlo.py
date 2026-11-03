"""Classical Monte Carlo baseline."""
from __future__ import annotations

import numpy as np

from .black_scholes import DiscreteModel


def mc_amplitude(model: DiscreteModel, num_samples: int, rng: np.random.Generator) -> float:
    """Monte Carlo estimate of a = E[f] on the same discrete grid the quantum circuit uses.

    Each sample is one draw of the distribution, i.e. one "query" of the classical oracle,
    so its cost is comparable with the number of A-operator calls in QAE.
    """
    idx = rng.choice(len(model.probs), size=num_samples, p=model.probs)
    return float(model.normalized_payoff[idx].mean())


def mc_call_price(S0, K, r, sigma, T, num_samples: int, rng: np.random.Generator) -> float:
    """Plain continuous Monte Carlo price under geometric Brownian motion."""
    z = rng.standard_normal(num_samples)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
    return float(np.exp(-r * T) * np.maximum(ST - K, 0.0).mean())
