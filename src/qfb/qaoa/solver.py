"""QAOA in PennyLane for any QUBO (portfolio selection, Max-Cut, ...)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from .qubo import QUBO, all_bitstrings, brute_force, qubo_to_ising


def ising_hamiltonian(qubo: QUBO) -> tuple[qml.operation.Operator, float]:
    h, J, offset = qubo_to_ising(qubo)
    coeffs, ops = [], []
    for i, hi in enumerate(h):
        if abs(hi) > 1e-12:
            coeffs.append(float(hi)); ops.append(qml.Z(i))
    n = qubo.n
    for i in range(n):
        for j in range(i + 1, n):
            if abs(J[i, j]) > 1e-12:
                coeffs.append(float(J[i, j])); ops.append(qml.Z(i) @ qml.Z(j))
    return qml.Hamiltonian(coeffs, ops), float(offset)


@dataclass
class QAOAResult:
    layers: int
    params: np.ndarray
    expected_energy: float      # <H> + offset at the optimized parameters
    approx_ratio: float         # (E_max - <E>) / (E_max - E_min), 1 is optimal
    p_optimal: float            # probability of sampling an optimal bitstring
    p_feasible: float           # probability of sampling a feasible bitstring (1 if no mask)
    most_likely: np.ndarray     # most probable bitstring
    probs: np.ndarray           # full output distribution (PennyLane ordering)
    history: list


def run_qaoa(
    qubo: QUBO, layers: int = 1, steps: int = 150, stepsize: float = 0.05,
    num_inits: int = 3, seed: int = 0, device: str = "lightning.qubit",
    mixer: str = "x", hamming_weight: int | None = None, feasible: np.ndarray | None = None,
) -> QAOAResult:
    """Optimize a depth-p QAOA with Adam from several random starts; keep the best.

    mixer="x":  standard transverse-field mixer, starts from |+>^n, constraint enforced only
                by the QUBO penalty.
    mixer="xy": ring XY mixer that preserves Hamming weight, started from the Dicke state with
                ``hamming_weight`` ones, so every sampled portfolio meets the budget. Use it with
                a penalty-free QUBO (``portfolio_qubo(..., penalty=0)``).
    feasible:   optional boolean mask over all 2^n bitstrings. If given, E_min / E_max / the
                optimal set are taken over feasible bitstrings only, and p_feasible is reported.
    """
    n = qubo.n
    wires = list(range(n))
    cost_h, offset = ising_hamiltonian(qubo)
    # Rescale the cost Hamiltonian so the same gamma range works across problem sizes.
    scale = max(1e-12, float(np.sum(np.abs(cost_h.terms()[0]))))
    cost_scaled = qml.Hamiltonian(np.array(cost_h.terms()[0]) / scale, cost_h.terms()[1])
    if mixer == "x":
        mixer_h = qml.qaoa.x_mixer(wires)
    elif mixer == "xy":
        import networkx as nx
        if hamming_weight is None:
            raise ValueError("mixer='xy' needs hamming_weight (the portfolio budget).")
        mixer_h = qml.qaoa.xy_mixer(nx.cycle_graph(n))
    else:
        raise ValueError(f"unknown mixer {mixer!r}")
    dev = qml.device(device, wires=n)
    if mixer == "xy":
        weights = all_bitstrings(n).sum(axis=1)
        dicke = (weights == hamming_weight).astype(float)
        dicke = dicke / np.linalg.norm(dicke)
    diff = "adjoint" if device.startswith("lightning") else "best"

    def layer(gamma, beta):
        qml.qaoa.cost_layer(gamma, cost_scaled)
        qml.qaoa.mixer_layer(beta, mixer_h)

    def ansatz(params):
        if mixer == "x":
            for w in wires:
                qml.Hadamard(w)
        else:
            # Dicke state: equal superposition of all bitstrings with `hamming_weight` ones.
            # Loaded as an amplitude vector for simulation; on hardware it would be built with
            # a Dicke-state circuit (Baertschi & Eidenbenz 2019).
            qml.StatePrep(dicke, wires=wires)
        qml.layer(layer, layers, params[0], params[1])

    @qml.qnode(dev, diff_method=diff)
    def energy(params):
        ansatz(params)
        return qml.expval(cost_scaled)

    @qml.qnode(dev)
    def probs(params):
        ansatz(params)
        return qml.probs(wires=wires)

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(num_inits):
        params = pnp.array(rng.uniform(0, np.pi, (2, layers)), requires_grad=True)
        opt = qml.AdamOptimizer(stepsize)
        hist = []
        for _ in range(steps):
            params, val = opt.step_and_cost(energy, params)
            hist.append(float(val) * scale + offset)
        final = float(energy(params)) * scale + offset
        if best is None or final < best[1]:
            best = (np.array(params), final, hist)

    params, exp_e, hist = best
    p = np.array(probs(params))
    _, _, _, energies = brute_force(qubo)
    mask = np.ones_like(energies, dtype=bool) if feasible is None else np.asarray(feasible, bool)
    e_min, e_max = float(energies[mask].min()), float(energies[mask].max())
    optimal = mask & np.isclose(energies, e_min, atol=1e-9)
    return QAOAResult(
        layers=layers,
        params=params,
        expected_energy=exp_e,
        approx_ratio=float((e_max - exp_e) / (e_max - e_min)) if e_max > e_min else 1.0,
        p_optimal=float(p[optimal].sum()),
        p_feasible=float(p[mask].sum()),
        most_likely=all_bitstrings(n)[int(np.argmax(p))],
        probs=p,
        history=hist,
    )
