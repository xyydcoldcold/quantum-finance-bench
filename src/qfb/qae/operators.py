"""The A (state preparation) and Q (Grover) operators for option pricing.

Register layout (Qiskit little-endian): qubits 0..n-1 hold the price index i,
qubit n is the objective qubit.

    A |0>_n |0>  =  sum_i sqrt(p_i) |i> ( sqrt(1 - f_i) |0> + sqrt(f_i) |1> )

so  P(objective = 1) = sum_i p_i f_i = E[f] = a,  with f = payoff / max payoff.

Q = -A S_0 A^dagger S_chi has eigenphases +-2*theta with a = sin^2(theta).
Quantum phase estimation on Q therefore recovers a: this is canonical
amplitude estimation (Brassard, Hoyer, Mosca, Tapp 2002).
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import StatePreparation, UCRYGate
from qiskit.quantum_info import Statevector

from ..finance.black_scholes import DiscreteModel


def build_state_prep(model: DiscreteModel) -> QuantumCircuit:
    """A operator: load the distribution, then rotate the objective qubit by the payoff."""
    n = int(np.log2(len(model.probs)))
    if 2**n != len(model.probs):
        raise ValueError("Grid size must be a power of two.")
    qc = QuantumCircuit(n + 1, name="A")
    qc.append(StatePreparation(np.sqrt(model.probs)), range(n))
    # Uniformly controlled RY: for price index i, RY(theta_i) with sin^2(theta_i / 2) = f_i.
    angles = 2.0 * np.arcsin(np.sqrt(np.clip(model.normalized_payoff, 0.0, 1.0)))
    qc.append(UCRYGate(list(angles)), [n, *range(n)])
    return qc


def build_grover(state_prep: QuantumCircuit) -> QuantumCircuit:
    """Q = -A S_0 A^dagger S_chi, where S_chi flips the sign of 'good' states (objective = 1)
    and S_0 = I - 2|0><0|. The explicit global phase pi makes controlled-Q correct inside QPE."""
    n_tot = state_prep.num_qubits
    obj = n_tot - 1
    qc = QuantumCircuit(n_tot, name="Q")
    # S_chi
    qc.z(obj)
    # A^dagger
    qc.compose(state_prep.inverse(), inplace=True)
    # S_0 = I - 2|0><0|
    qc.x(range(n_tot))
    if n_tot == 1:
        qc.z(0)
    else:
        qc.h(obj)
        qc.mcx(list(range(n_tot - 1)), obj)
        qc.h(obj)
    qc.x(range(n_tot))
    # A
    qc.compose(state_prep, inplace=True)
    qc.global_phase += np.pi
    return qc


def objective_probability(circuit: QuantumCircuit) -> float:
    """Exact probability that the last qubit (objective) is measured as 1."""
    sv = Statevector(circuit)
    return float(sv.probabilities([circuit.num_qubits - 1])[1])
