# Quantum Finance Bench

Quantum algorithms for two finance problems, benchmarked against classical baselines:

| Module | Problem | Quantum method | Framework | Classical baselines |
|---|---|---|---|---|
| A | European option pricing | Canonical QAE (quantum phase estimation on the Grover operator), maximum-likelihood AE | Qiskit | Monte Carlo, Black-Scholes closed form |
| B | Mean-variance portfolio selection | QAOA on a QUBO / Ising Hamiltonian, X mixer + penalty vs Hamming-weight-preserving XY mixer | PennyLane | Exact enumeration, simulated annealing |
| B | Asset clustering (graph theory) | QAOA Max-Cut on a correlation-distance graph | PennyLane | Exact, one-exchange local search |

All results in `results/` are produced by the scripts in `experiments/`; nothing is hand-entered.

## Setup

```bash
conda env create -f environment.yml
conda activate qfb
pytest
```

`environment.yml` installs this package in editable mode (`pip install -e .`).

## Run the experiments

```bash
cd experiments
python 01_qae_convergence.py --trials 20 --max-eval-qubits 9 --max-powers 9   # error vs oracle calls
python 02_qae_resources.py --price-qubits 2 3 4 --max-eval-qubits 5    # CNOTs / depth vs precision
python 03_portfolio_qaoa.py --assets 4 6 8 --layers 1 2 3              # QAOA vs SA vs exact
python 04_maxcut_clustering.py --assets 8 --layers 1 2 3               # Max-Cut clustering
```

Each script writes CSV and PNG files to `results/`. Runtimes on a laptop are seconds to a few
minutes; `--assets 10` or more makes QAOA simulation noticeably slower.

Use real market data by exporting a CSV of daily prices (rows = dates, columns = tickers) and
passing `--prices-csv path/to/prices.csv` to scripts 03 and 04.

## Module A: option pricing with quantum amplitude estimation

1. `finance/black_scholes.py` discretizes the risk-neutral log-normal distribution of S_T onto
   2^n grid points (equal bins over mean +/- 5 std, exact bin probabilities from the CDF), and
   gives the Black-Scholes reference price. With the default 5 price qubits the grid price is
   within about 0.5% of Black-Scholes.
2. `qae/operators.py` builds
   * **A**: `StatePreparation` loads sqrt(p_i) on n price qubits, then a uniformly controlled
     RY sets P(objective = 1 | i) = f_i = payoff_i / max payoff. So P(objective = 1) = a = E[f].
   * **Q = -A S_0 A^dagger S_chi**, whose eigenphases are +-2 theta with a = sin^2(theta).
3. `qae/canonical.py` runs **quantum phase estimation on Q** with m evaluation qubits
   (controlled Q^(2^j) followed by an inverse QFT); the outcome y gives a = sin^2(pi y / 2^m).
   Error scales as O(1/M), M = 2^m, versus O(1/sqrt(N)) for Monte Carlo.
4. `qae/mlae.py` implements **maximum-likelihood amplitude estimation** (Suzuki et al. 2020): no
   QPE register, circuits Q^k A for k = 0, 1, 2, 4, ..., and a likelihood fit for theta. This is
   the variant that suits near-term hardware.
5. `qae/resources.py` transpiles the real gate-level circuits to {CX, RZ, SX, X} and reports
   qubits, CNOT count, and depth (the same resource-estimation mindset as fault-tolerant costing,
   applied to a finance workload).

Oracle calls are counted as applications of A or A^dagger: a shot with k Grover iterations
costs 2k + 1 calls. Canonical QAE uses 10 shots per run by default (`--qae-shots`) and keeps
the most frequent outcome. Its exact outcome distribution is computed from the spectrum of Q
(`qpe_distribution_eigen`, a Fejer-kernel formula) and cross-checked against full circuit
simulation in the tests. For simulation speed, `canonical_qae` and `mlae` use dense matrices for
Q^(2^j); `canonical_qae_circuit(..., use_unitary=False)` keeps the gate-level decomposition.

Things worth discussing in results: canonical QAE estimates are restricted to the grid
sin^2(pi y / M), so the error can plateau when the true a sits between grid points; MLAE
avoids that discretization. The grid price also differs from Black-Scholes because of the
2^n-point discretization, which is a separate error source from the estimator.

## Module B: QAOA for portfolio optimization and Max-Cut

* `qaoa/qubo.py`: QUBO energy, QUBO to Ising mapping (x = (1 - z) / 2), brute force, and
  single-bit-flip simulated annealing.
* `qaoa/portfolio.py`: min q x^T Sigma x - mu^T x + lambda (sum x - B)^2, pick B of n assets.
* `qaoa/maxcut.py`: Mantegna distance d_ij = sqrt(2 (1 - rho_ij)); a maximum cut minimizes
  within-group distance, so it splits assets into two internally correlated groups.
* `qaoa/solver.py`: PennyLane QAOA (`qml.qaoa.cost_layer` / `mixer_layer`), Adam with several
  random starts, `lightning.qubit` with adjoint differentiation.
  * `mixer="x"`: standard mixer from |+>^n, budget enforced by the penalty term.
  * `mixer="xy"`: ring XY mixer from the Dicke state, so every sample satisfies the budget
    and no penalty is needed (the Dicke state is loaded as an amplitude vector in simulation).

Metrics: P(optimal) is the probability of sampling the optimal portfolio and is comparable
across mixers. The approximation ratio (E_max - <E>) / (E_max - E_min) is taken over all
bitstrings for the X mixer (penalized infeasible states inflate it) and over feasible
bitstrings for the XY mixer, so compare ratios only within one mixer.

## Layout

```
src/qfb/
  data.py                 synthetic sector-factor returns, CSV loader, annualized mu / Sigma
  finance/                Black-Scholes, discretized log-normal model, Monte Carlo
  qae/                    A and Q operators, canonical QAE, MLAE, resource counts
  qaoa/                   QUBO tools, portfolio and Max-Cut encodings, PennyLane QAOA
experiments/              01-04 scripts, outputs to results/
tests/                    pytest checks (state prep, QAE accuracy, Ising mapping, SA, QAOA)
```

## References

* G. Brassard, P. Hoyer, M. Mosca, A. Tapp. Quantum amplitude amplification and estimation. 2002.
* S. Woerner, D. J. Egger. Quantum risk analysis. npj Quantum Information 5, 15 (2019).
* N. Stamatopoulos et al. Option pricing using quantum computers. Quantum 4, 291 (2020).
* Y. Suzuki et al. Amplitude estimation without phase estimation. Quantum Inf. Process. 19, 75 (2020).
* E. Farhi, J. Goldstone, S. Gutmann. A quantum approximate optimization algorithm. arXiv:1411.4028.
* S. Hadfield et al. From the QAOA to a quantum alternating operator ansatz. Algorithms 12, 34 (2019).
* A. Baertschi, S. Eidenbenz. Deterministic preparation of Dicke states. 2019.
* R. N. Mantegna. Hierarchical structure in financial markets. Eur. Phys. J. B 11, 193 (1999).
