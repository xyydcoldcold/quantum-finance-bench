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

1. `finance/black_scholes.py` discretizes the risk-neutral log-normal distribution of $`S_T`$ onto
   $`2^n`$ grid points (equal bins over mean $`\pm 5`$ std, exact bin probabilities from the CDF), and
   gives the Black-Scholes reference price. With the default 5 price qubits the grid price is
   within about 0.5% of Black-Scholes.
2. `qae/operators.py` builds the two operators used by every estimator:

```math
\mathcal{A}\,|0\rangle_n|0\rangle = \sum_i \sqrt{p_i}\,|i\rangle\left(\sqrt{1-f_i}\,|0\rangle + \sqrt{f_i}\,|1\rangle\right),
\qquad f_i = \frac{\max(S_i - K,\,0)}{\max_j \max(S_j - K,\,0)}
```

   `StatePreparation` loads $`\sqrt{p_i}`$ on the $`n`$ price qubits and a uniformly controlled
   $`R_Y`$ rotates the objective qubit, so $`P(\text{objective}=1) = a = \mathbb{E}[f]`$ and the
   option price is $`e^{-rT}\,\max_j(S_j-K)^+\,a`$. The Grover operator is

```math
\mathcal{Q} = -\,\mathcal{A}\,S_0\,\mathcal{A}^\dagger S_\chi, \qquad
\text{eigenphases } \pm 2\theta, \qquad a = \sin^2\theta .
```

3. `qae/canonical.py` runs **quantum phase estimation on $`\mathcal{Q}`$** with $`m`$ evaluation qubits
   (controlled $`\mathcal{Q}^{2^j}`$ followed by an inverse QFT). An outcome $`y`$ gives
   $`\hat a = \sin^2(\pi y / M)`$ with $`M = 2^m`$, so the error scales as $`O(1/M)`$ versus
   $`O(1/\sqrt{N})`$ for Monte Carlo.
4. `qae/mlae.py` implements **maximum-likelihood amplitude estimation** (Suzuki et al. 2020): no
   QPE register, only circuits $`\mathcal{Q}^{k}\mathcal{A}`$ for $`k = 0, 1, 2, 4, \dots`$ and a fit of

```math
\hat\theta = \arg\max_\theta \prod_k \sin^{2h_k}\!\big((2k+1)\theta\big)\,\cos^{2(N-h_k)}\!\big((2k+1)\theta\big),
```

   where $`h_k`$ is the number of "good" outcomes out of $`N`$ shots. This is the variant that
   suits near-term hardware.
5. `qae/resources.py` transpiles the real gate-level circuits to {CX, RZ, SX, X} and reports
   qubits, CNOT count, and depth (the same resource-estimation mindset as fault-tolerant costing,
   applied to a finance workload).

Oracle calls are counted as applications of $`\mathcal{A}`$ or $`\mathcal{A}^\dagger`$: a shot with
$`k`$ Grover iterations costs $`2k+1`$ calls. Canonical QAE uses 10 shots per run by default
(`--qae-shots`) and keeps the most frequent outcome. Its exact outcome distribution is computed from
the spectrum of $`\mathcal{Q}`$ (`qpe_distribution_eigen`): writing
$`\mathcal{A}|0\rangle = \sum_j c_j |v_j\rangle`$ with $`\mathcal{Q}|v_j\rangle = e^{2\pi i \varphi_j}|v_j\rangle`$,

```math
P(y) = \sum_j |c_j|^2 \left| \frac{1}{M} \sum_{k=0}^{M-1} e^{2\pi i k (\varphi_j - y/M)} \right|^2 ,
```

which the tests cross-check against full circuit simulation. For simulation speed, `canonical_qae`
and `mlae` use dense matrices for $`\mathcal{Q}^{2^j}`$; `canonical_qae_circuit(..., use_unitary=False)`
keeps the gate-level decomposition.

Things worth discussing in results: canonical QAE estimates are restricted to the grid
$`\sin^2(\pi y/M)`$, so the error can plateau when the true $`a`$ sits between grid points; MLAE
avoids that discretization. The grid price also differs from Black-Scholes because of the
$`2^n`$-point discretization, which is a separate error source from the estimator.

## Module B: QAOA for portfolio optimization and Max-Cut

* `qaoa/qubo.py`: QUBO energy, QUBO to Ising mapping $`x_i = (1 - z_i)/2`$, brute force, and
  single-bit-flip simulated annealing.
* `qaoa/portfolio.py`: pick $`B`$ of $`n`$ assets ($`x_i = 1`$ means asset $`i`$ is held) by minimizing

```math
E(x) = q\,x^\top \Sigma\, x \;-\; \mu^\top x \;+\; \lambda \Big(\sum_i x_i - B\Big)^2 .
```

* `qaoa/maxcut.py`: Mantegna distance $`d_{ij} = \sqrt{2(1-\rho_{ij})}`$; a maximum cut minimizes
  within-group distance, so it splits assets into two internally correlated groups.
* `qaoa/solver.py`: PennyLane QAOA (`qml.qaoa.cost_layer` / `mixer_layer`), Adam with several
  random starts, `lightning.qubit` with adjoint differentiation.
  * `mixer="x"`: standard mixer from $`|+\rangle^{\otimes n}`$, budget enforced by the penalty term.
  * `mixer="xy"`: ring XY mixer from the Dicke state $`|D^n_B\rangle`$, so every sample satisfies
    the budget and no penalty is needed (the Dicke state is loaded as an amplitude vector in simulation).

Metrics: $`P(\text{optimal})`$ is the probability of sampling the optimal portfolio and is comparable
across mixers. The approximation ratio $`(E_{\max} - \langle E\rangle)/(E_{\max} - E_{\min})`$ is taken
over all bitstrings for the X mixer (penalized infeasible states inflate it) and over feasible
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

* G. Brassard, P. Hoyer, M. Mosca, A. Tapp. Quantum amplitude amplification and estimation. 2002. [arXiv:quant-ph/0005055](https://arxiv.org/abs/quant-ph/0005055)
* S. Woerner, D. J. Egger. Quantum risk analysis. npj Quantum Information 5, 15 (2019). [arXiv:1806.06893](https://arxiv.org/abs/1806.06893)
* N. Stamatopoulos et al. Option pricing using quantum computers. Quantum 4, 291 (2020). [arXiv:1905.02666](https://arxiv.org/abs/1905.02666)
* Y. Suzuki et al. Amplitude estimation without phase estimation. Quantum Inf. Process. 19, 75 (2020). [arXiv:1904.10246](https://arxiv.org/abs/1904.10246)
* E. Farhi, J. Goldstone, S. Gutmann. A quantum approximate optimization algorithm. 2014. [arXiv:1411.4028](https://arxiv.org/abs/1411.4028)
* S. Hadfield et al. From the quantum approximate optimization algorithm to a quantum alternating operator ansatz. Algorithms 12, 34 (2019). [arXiv:1709.03489](https://arxiv.org/abs/1709.03489)
* A. Baertschi, S. Eidenbenz. Deterministic preparation of Dicke states. 2019. [arXiv:1904.07358](https://arxiv.org/abs/1904.07358)
* R. N. Mantegna. Hierarchical structure in financial markets. Eur. Phys. J. B 11, 193 (1999). [arXiv:cond-mat/9802256](https://arxiv.org/abs/cond-mat/9802256)
