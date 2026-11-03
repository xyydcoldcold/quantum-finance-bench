"""Maximum-likelihood amplitude estimation (Suzuki et al., 2020).

QPE-free variant suited to near-term hardware: run Q^{m_k} A for a schedule
m_k = 0, 1, 2, 4, ..., measure only the objective qubit, and maximize
    L(theta) = prod_k  sin^2((2 m_k + 1) theta)^{h_k} cos^2((2 m_k + 1) theta)^{N - h_k}.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector
from scipy.optimize import minimize_scalar


@dataclass
class MLAEResult:
    estimate: float
    schedule: list[int]
    shots: int
    hits: list[int]
    oracle_calls: int


def _good_probability(state_prep: QuantumCircuit, q_mat: np.ndarray, power: int) -> float:
    sv = Statevector(state_prep).evolve(Operator(np.linalg.matrix_power(q_mat, power)))
    return float(sv.probabilities([state_prep.num_qubits - 1])[1])


def mlae(
    state_prep: QuantumCircuit,
    grover: QuantumCircuit,
    num_powers: int,
    shots: int = 100,
    rng: np.random.Generator | None = None,
) -> MLAEResult:
    rng = rng or np.random.default_rng()
    schedule = [0] + [2**k for k in range(num_powers - 1)]
    q_mat = Operator(grover).data
    hits = [int(rng.binomial(shots, _good_probability(state_prep, q_mat, m))) for m in schedule]

    ms = np.array(schedule)
    hs = np.array(hits)

    def neg_log_lik(theta: float) -> float:
        ang = (2 * ms + 1) * theta
        p1 = np.clip(np.sin(ang) ** 2, 1e-15, 1 - 1e-15)
        return -float(np.sum(hs * np.log(p1) + (shots - hs) * np.log(1 - p1)))

    # Global grid search, then local refinement around the best grid point.
    grid = np.linspace(0, np.pi / 2, 20000)
    vals = np.array([neg_log_lik(t) for t in grid])
    t0 = grid[np.argmin(vals)]
    step = grid[1] - grid[0]
    res = minimize_scalar(neg_log_lik, bounds=(max(0, t0 - step), min(np.pi / 2, t0 + step)), method="bounded")
    theta = float(res.x)
    oracle_calls = int(sum(shots * (2 * m + 1) for m in schedule))
    return MLAEResult(estimate=float(np.sin(theta) ** 2), schedule=schedule, shots=shots, hits=hits, oracle_calls=oracle_calls)
