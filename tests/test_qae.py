import numpy as np

from qfb.finance.black_scholes import bs_call_price, lognormal_grid
from qfb.qae import (build_grover, build_state_prep, canonical_qae, mlae, objective_probability,
                     qpe_distribution_circuit, qpe_distribution_eigen)

OPT = dict(S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0)


def test_state_prep_encodes_expected_payoff():
    model = lognormal_grid(**OPT, num_qubits=3)
    A = build_state_prep(model)
    assert np.isclose(objective_probability(A), model.exact_amplitude)


def test_grid_price_close_to_black_scholes():
    model = lognormal_grid(**OPT, num_qubits=5)
    assert abs(model.exact_price - bs_call_price(**OPT)) / bs_call_price(**OPT) < 0.01


def test_canonical_qae_within_resolution():
    model = lognormal_grid(**OPT, num_qubits=3)
    A = build_state_prep(model)
    Q = build_grover(A)
    m = 6
    r = canonical_qae(A, Q, m, shots=200, rng=np.random.default_rng(0))
    # Resolution of canonical QAE: |a_hat - a| <= pi/M + (pi/M)^2 with high probability.
    M = 2**m
    assert abs(r.estimate - model.exact_amplitude) <= np.pi / M + (np.pi / M) ** 2


def test_mlae_converges():
    model = lognormal_grid(**OPT, num_qubits=3)
    A = build_state_prep(model)
    Q = build_grover(A)
    r = mlae(A, Q, 6, shots=200, rng=np.random.default_rng(0))
    assert abs(r.estimate - model.exact_amplitude) < 0.01


def test_eigen_qpe_distribution_matches_circuit():
    model = lognormal_grid(**OPT, num_qubits=3)
    A = build_state_prep(model)
    Q = build_grover(A)
    for m in (3, 5):
        assert np.allclose(qpe_distribution_eigen(A, Q, m), qpe_distribution_circuit(A, Q, m), atol=1e-8)
