"""Canonical quantum amplitude estimation = quantum phase estimation on the Grover operator Q.

With m evaluation qubits (M = 2**m) the measured integer y gives
    a_hat = sin^2(pi * y / M),
and the estimation error scales as O(1/M), versus O(1/sqrt(N)) for Monte Carlo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import QFTGate, UnitaryGate
from qiskit.quantum_info import Operator, Statevector


@dataclass
class QAEResult:
    estimate: float                 # a_hat from the most frequent outcome
    num_eval_qubits: int
    shots: int
    grover_calls_per_shot: int      # M - 1 applications of Q
    oracle_calls: int               # total calls of A or A^dagger over all shots
    distribution: dict              # a_hat value -> exact probability


def canonical_qae_circuit(
    state_prep: QuantumCircuit, grover: QuantumCircuit, num_eval_qubits: int, use_unitary: bool = True
) -> QuantumCircuit:
    """Build the QPE circuit. Evaluation qubits come first, then the state register.

    ``use_unitary=True`` packs Q^(2^j) into a dense UnitaryGate, which is fast and exact for
    simulation of small registers. ``use_unitary=False`` keeps the real gate-level
    decomposition, which is what ``circuit_resources`` counts.
    """
    m, n_tot = num_eval_qubits, state_prep.num_qubits
    qc = QuantumCircuit(m + n_tot, name=f"QAE(m={m})")
    state = list(range(m, m + n_tot))
    qc.compose(state_prep, state, inplace=True)
    qc.h(range(m))
    if use_unitary:
        q_mat = Operator(grover).data
    for j in range(m):
        power = 2**j
        if use_unitary:
            gate = UnitaryGate(np.linalg.matrix_power(q_mat, power), label=f"Q^{power}").control(1)
        else:
            gate = grover.power(power).to_gate(label=f"Q^{power}").control(1)
        qc.append(gate, [j, *state])
    qc.append(QFTGate(m).inverse(), range(m))
    return qc


def qpe_distribution_eigen(state_prep: QuantumCircuit, grover: QuantumCircuit, num_eval_qubits: int) -> np.ndarray:
    """Exact QPE outcome distribution from the spectrum of Q (no circuit simulation).

    Write A|0> = sum_j c_j |v_j> with Q|v_j> = exp(2 pi i phi_j)|v_j>. QPE with M = 2^m
    evaluation qubits then returns y with probability
        P(y) = sum_j |c_j|^2 * | (1/M) sum_{k<M} exp(2 pi i k (phi_j - y/M)) |^2   (Fejer kernel).
    Q is unitary and here normal, so its eigenvectors are orthonormal.
    """
    M = 2**num_eval_qubits
    q_mat = Operator(grover).data
    evals, evecs = np.linalg.eig(q_mat)
    # Re-orthonormalize degenerate eigenspaces (QR), which eig does not guarantee.
    evecs, _ = np.linalg.qr(evecs)
    psi = Statevector(state_prep).data
    weights = np.abs(evecs.conj().T @ psi) ** 2
    phis = np.angle(evals) / (2 * np.pi)
    k = np.arange(M)
    y = np.arange(M)
    probs = np.zeros(M)
    for w, phi in zip(weights, phis):
        if w < 1e-14:
            continue
        amp = np.exp(2j * np.pi * np.outer(phi - y / M, k)).mean(axis=1)
        probs += w * np.abs(amp) ** 2
    return probs / probs.sum()


def qpe_distribution_circuit(state_prep: QuantumCircuit, grover: QuantumCircuit, num_eval_qubits: int) -> np.ndarray:
    """Same distribution by simulating the full QPE circuit (slower; used to cross-check)."""
    qc = canonical_qae_circuit(state_prep, grover, num_eval_qubits, use_unitary=True)
    return Statevector(qc).probabilities(list(range(num_eval_qubits)))


def canonical_qae(
    state_prep: QuantumCircuit,
    grover: QuantumCircuit,
    num_eval_qubits: int,
    shots: int = 100,
    rng: np.random.Generator | None = None,
    method: str = "eigen",
    probs_y: np.ndarray | None = None,
) -> QAEResult:
    """Run canonical QAE. ``method`` picks how the exact outcome distribution is computed
    ("eigen": spectral formula, "circuit": statevector simulation of the QPE circuit); shots
    are then sampled from it. Pass ``probs_y`` to reuse a distribution across repeated trials."""
    rng = rng or np.random.default_rng()
    m = num_eval_qubits
    M = 2**m
    if probs_y is None:
        fn = qpe_distribution_eigen if method == "eigen" else qpe_distribution_circuit
        probs_y = fn(state_prep, grover, m)  # index y = integer on eval register

    # Map outcomes y -> a_hat, merging y and M - y (they give the same amplitude).
    a_vals = np.round(np.sin(np.pi * np.arange(M) / M) ** 2, 12)
    dist: dict[float, float] = {}
    for a, p in zip(a_vals, probs_y):
        dist[float(a)] = dist.get(float(a), 0.0) + float(p)

    samples = rng.choice(M, size=shots, p=probs_y / probs_y.sum())
    values, counts = np.unique(a_vals[samples], return_counts=True)
    estimate = float(values[np.argmax(counts)])
    grover_calls = M - 1
    return QAEResult(
        estimate=estimate,
        num_eval_qubits=m,
        shots=shots,
        grover_calls_per_shot=grover_calls,
        oracle_calls=shots * (2 * grover_calls + 1),
        distribution=dist,
    )
