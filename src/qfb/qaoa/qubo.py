"""Generic QUBO utilities: energy, Ising mapping, exact and simulated-annealing baselines.

QUBO:   E(x) = x^T Q x + c^T x + const,   x in {0,1}^n   (Q symmetric)
Ising:  x_i = (1 - z_i) / 2,  z_i in {+1,-1}  ->  H = sum_i h_i Z_i + sum_{i<j} J_ij Z_i Z_j + offset
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class QUBO:
    Q: np.ndarray
    c: np.ndarray
    const: float = 0.0

    def __post_init__(self):
        self.Q = 0.5 * (np.asarray(self.Q, float) + np.asarray(self.Q, float).T)
        self.c = np.asarray(self.c, float)

    @property
    def n(self) -> int:
        return len(self.c)

    def energy(self, x: np.ndarray) -> np.ndarray:
        """Energy of one bitstring (shape n) or many (shape k x n)."""
        x = np.atleast_2d(x).astype(float)
        e = np.einsum("ki,ij,kj->k", x, self.Q, x) + x @ self.c + self.const
        return e if e.shape[0] > 1 else e[0]


def qubo_to_ising(qubo: QUBO) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (h, J, offset) with J upper triangular."""
    Q, c, n = qubo.Q, qubo.c, qubo.n
    h = np.zeros(n)
    J = np.zeros((n, n))
    offset = qubo.const
    for i in range(n):
        # Q_ii x_i^2 = Q_ii x_i, plus linear term c_i x_i
        lin = Q[i, i] + c[i]
        offset += lin / 2
        h[i] -= lin / 2
        for j in range(i + 1, n):
            w = 2 * Q[i, j]  # x^T Q x counts (i,j) and (j,i)
            offset += w / 4
            h[i] -= w / 4
            h[j] -= w / 4
            J[i, j] += w / 4
    return h, J, offset


def all_bitstrings(n: int) -> np.ndarray:
    """All 2^n bitstrings, row k = binary of k with x[0] as the most significant bit
    (the same ordering as PennyLane's qml.probs on wires 0..n-1)."""
    return np.array(list(product([0, 1], repeat=n)), dtype=int)


def brute_force(qubo: QUBO) -> tuple[np.ndarray, float, float, np.ndarray]:
    """Exact solution by enumeration. Returns (best_x, E_min, E_max, energies of all states)."""
    xs = all_bitstrings(qubo.n)
    e = qubo.energy(xs)
    k = int(np.argmin(e))
    return xs[k], float(e[k]), float(e.max()), e


def simulated_annealing(
    qubo: QUBO, num_sweeps: int = 2000, t_start: float | None = None, t_end: float = 1e-3,
    num_restarts: int = 10, rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, float]:
    """Classical (quantum-inspired baseline) single-bit-flip simulated annealing."""
    rng = rng or np.random.default_rng()
    n, Q, c = qubo.n, qubo.Q, qubo.c
    if t_start is None:
        t_start = max(1e-9, float(np.abs(Q).sum(axis=1).max() + np.abs(c).max()))
    temps = np.geomspace(t_start, t_end, num_sweeps)
    best_x, best_e = None, np.inf
    for _ in range(num_restarts):
        x = rng.integers(0, 2, n).astype(float)
        e = float(qubo.energy(x))
        for T in temps:
            for i in rng.permutation(n):
                d = 1 - 2 * x[i]  # +1 if flipping 0->1, -1 if 1->0
                # Delta E = d * (2 (Qx)_i + c_i) + Q_ii d^2
                dE = d * (2 * (Q[i] @ x) + c[i]) + Q[i, i] * d * d
                if dE <= 0 or rng.random() < np.exp(-dE / T):
                    x[i] += d
                    e += dE
            if e < best_e:
                best_e, best_x = e, x.copy()
    return best_x.astype(int), float(qubo.energy(best_x))
