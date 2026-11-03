"""Module A: option pricing error vs number of oracle calls.

Compares canonical QAE (QPE on Q), maximum-likelihood AE, and classical Monte Carlo on the
same discretized distribution. Error is |a_hat - a| (amplitude), and price error =
discount * max_payoff * amplitude error. Expected scaling: QAE ~ 1/N, MC ~ 1/sqrt(N).

    python experiments/01_qae_convergence.py --price-qubits 5 --trials 20
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import OPTION, RESULTS

from qfb.finance.black_scholes import bs_call_price, lognormal_grid
from qfb.finance.monte_carlo import mc_amplitude
from qfb.qae import build_grover, build_state_prep, canonical_qae, mlae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-qubits", type=int, default=5)
    ap.add_argument("--max-eval-qubits", type=int, default=8)
    ap.add_argument("--max-powers", type=int, default=8)
    ap.add_argument("--shots", type=int, default=100, help="shots per MLAE circuit")
    ap.add_argument("--qae-shots", type=int, default=10,
                    help="shots per canonical QAE run (the most frequent outcome is kept)")
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    model = lognormal_grid(**OPTION, num_qubits=args.price_qubits)
    A = build_state_prep(model)
    Q = build_grover(A)
    a = model.exact_amplitude
    to_price = model.discount * model.payoff_scale
    print(f"Black-Scholes price      : {bs_call_price(**OPTION):.4f}")
    print(f"Grid price ({2**args.price_qubits} points)   : {model.exact_price:.4f}  (a = {a:.6f})")
    print("Errors below are measured against the grid price; the grid vs Black-Scholes gap is "
          "discretization error, which shrinks with --price-qubits.")

    from qfb.qae import qpe_distribution_eigen
    qpe_probs = {m: qpe_distribution_eigen(A, Q, m) for m in range(1, args.max_eval_qubits + 1)}
    rows = []
    for t in range(args.trials):
        for m in range(1, args.max_eval_qubits + 1):
            r = canonical_qae(A, Q, m, shots=args.qae_shots, rng=rng, probs_y=qpe_probs[m])
            rows.append(dict(method="Canonical QAE", trial=t, param=m, oracle_calls=r.oracle_calls,
                             estimate=r.estimate))
        for k in range(1, args.max_powers + 1):
            r = mlae(A, Q, k, shots=args.shots, rng=rng)
            rows.append(dict(method="MLAE", trial=t, param=k, oracle_calls=r.oracle_calls, estimate=r.estimate))
        for n in np.unique(np.logspace(2, 6, 13).astype(int)):
            rows.append(dict(method="Monte Carlo", trial=t, param=int(n), oracle_calls=int(n),
                             estimate=mc_amplitude(model, int(n), rng)))

    df = pd.DataFrame(rows)
    df["abs_error"] = (df["estimate"] - a).abs()
    df["price_error"] = df["abs_error"] * to_price
    df.to_csv(RESULTS / "qae_convergence_raw.csv", index=False)
    summary = df.groupby(["method", "param"]).agg(
        oracle_calls=("oracle_calls", "mean"), mean_abs_error=("abs_error", "mean"),
        median_abs_error=("abs_error", "median"), mean_price_error=("price_error", "mean")).reset_index()
    summary.to_csv(RESULTS / "qae_convergence_summary.csv", index=False)
    print(summary.to_string(index=False))

    # Fitted slopes on log-log axes (QAE theory: -1, MC theory: -0.5).
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, g in summary.groupby("method"):
        g = g[g["mean_abs_error"] > 0]
        ax.loglog(g["oracle_calls"], g["mean_abs_error"], "o-", label=method)
        if len(g) >= 3:
            slope = np.polyfit(np.log(g["oracle_calls"]), np.log(g["mean_abs_error"]), 1)[0]
            print(f"{method:14s} fitted slope: {slope:.2f}")
    n = np.logspace(2, 6, 50)
    ax.loglog(n, 0.5 / np.sqrt(n), "k:", lw=1, label=r"$O(1/\sqrt{N})$")
    ax.loglog(n, 5.0 / n, "k--", lw=1, label=r"$O(1/N)$")
    ax.set_xlabel("oracle calls (applications of A or A$^\\dagger$)")
    ax.set_ylabel("mean |estimated amplitude - exact|")
    ax.set_title(f"European call pricing, {2**args.price_qubits}-point grid")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "qae_convergence.png", dpi=200)
    print(f"Saved results to {RESULTS}")


if __name__ == "__main__":
    main()
