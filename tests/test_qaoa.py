import numpy as np
import pennylane as qml

from qfb.data import annualized_stats, synthetic_returns
from qfb.qaoa import brute_force, maxcut_qubo, correlation_distance_graph, portfolio_qubo, run_qaoa, simulated_annealing
from qfb.qaoa.maxcut import cut_value
from qfb.qaoa.portfolio import is_feasible
from qfb.qaoa.qubo import all_bitstrings
from qfb.qaoa.solver import ising_hamiltonian


def _problem(n=5):
    mu, sigma = annualized_stats(synthetic_returns(n, seed=1))
    return portfolio_qubo(mu, sigma, budget=2)


def test_ising_matches_qubo_energies():
    q = _problem()
    H, offset = ising_hamiltonian(q)
    diag = np.real(np.diag(qml.matrix(H, wire_order=range(q.n)))) + offset
    assert np.allclose(diag, q.energy(all_bitstrings(q.n)))


def test_penalty_makes_optimum_feasible():
    q = _problem()
    x, *_ = brute_force(q)
    assert x.sum() == 2


def test_simulated_annealing_finds_optimum():
    q = _problem()
    _, e_min, _, _ = brute_force(q)
    _, e_sa = simulated_annealing(q, num_sweeps=300, rng=np.random.default_rng(0))
    assert np.isclose(e_sa, e_min)


def test_maxcut_qubo_is_negative_cut():
    g = correlation_distance_graph(synthetic_returns(5, seed=2))
    q = maxcut_qubo(g)
    for x in all_bitstrings(5):
        assert np.isclose(-q.energy(x), cut_value(g, x))


def test_qaoa_beats_random_and_xy_stays_feasible():
    q = _problem()
    r = run_qaoa(q, layers=1, steps=40, num_inits=1)
    assert r.approx_ratio > 0.5
    mu, sigma = annualized_stats(synthetic_returns(5, seed=1))
    q0 = portfolio_qubo(mu, sigma, budget=2, penalty=0.0)
    mask = is_feasible(all_bitstrings(5), 2)
    r = run_qaoa(q0, layers=1, steps=40, num_inits=1, mixer="xy", hamming_weight=2, feasible=mask)
    assert np.isclose(r.p_feasible, 1.0)
