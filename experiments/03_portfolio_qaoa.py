"""Module B: mean-variance portfolio selection with QAOA (PennyLane).

For each problem size n and depth p, compares
  * QAOA with the standard X mixer + budget penalty
  * QAOA with a Hamming-weight-preserving XY mixer (no penalty needed)
  * classical simulated annealing and exact enumeration.

    python experiments/03_portfolio_qaoa.py --assets 4 6 8 --layers 1 2 3
    python experiments/03_portfolio_qaoa.py --prices-csv my_prices.csv --assets 6
"""
import argparse
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from common import RESULTS

from qfb.data import annualized_stats, returns_from_price_csv, synthetic_returns
from qfb.qaoa import brute_force, portfolio_qubo, run_qaoa, simulated_annealing
from qfb.qaoa.portfolio import is_feasible
from qfb.qaoa.qubo import all_bitstrings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=int, nargs="+", default=[4, 6, 8])
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--risk-aversion", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--inits", type=int, default=2)
    ap.add_argument("--prices-csv", default=None, help="optional CSV of prices (dates x tickers)")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rows = []
    for n in args.assets:
        if args.prices_csv:
            rets = returns_from_price_csv(args.prices_csv).iloc[:, :n]
        else:
            rets = synthetic_returns(n, seed=args.seed)
        mu, sigma = annualized_stats(rets)
        budget = n // 2
        q_pen = portfolio_qubo(mu, sigma, budget, args.risk_aversion)
        q_free = portfolio_qubo(mu, sigma, budget, args.risk_aversion, penalty=0.0)
        mask = is_feasible(all_bitstrings(n), budget)
        x_opt, e_opt, _, _ = brute_force(q_pen)
        x_sa, e_sa = simulated_annealing(q_pen, num_sweeps=500, rng=np.random.default_rng(args.seed))
        print(f"\nn={n}, budget={budget}: exact {x_opt} E={e_opt:.4f} | SA {x_sa} E={e_sa:.4f}")
        rows.append(dict(assets=n, method="Simulated annealing", layers=0, approx_ratio=np.nan,
                         p_optimal=np.nan, p_feasible=np.nan, found_optimum=bool(np.isclose(e_sa, e_opt)),
                         seconds=np.nan))
        for p in args.layers:
            for label, kw, qubo in [
                ("QAOA X mixer + penalty", dict(mixer="x"), q_pen),
                ("QAOA XY mixer", dict(mixer="xy", hamming_weight=budget), q_free),
            ]:
                t0 = time.time()
                r = run_qaoa(qubo, layers=p, steps=args.steps, num_inits=args.inits, seed=args.seed,
                             feasible=mask if kw["mixer"] == "xy" else None, **kw)
                p_feas = float(r.probs[mask].sum())
                rows.append(dict(assets=n, method=label, layers=p, approx_ratio=r.approx_ratio,
                                 p_optimal=r.p_optimal, p_feasible=p_feas,
                                 found_optimum=bool(np.array_equal(r.most_likely, x_opt)),
                                 seconds=time.time() - t0))
                print(f"  p={p} {label:24s} ratio={r.approx_ratio:.3f} P(opt)={r.p_optimal:.3f} "
                      f"P(feasible)={p_feas:.3f} argmax={r.most_likely}")

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "portfolio_qaoa.csv", index=False)

    q = df[df.method.str.startswith("QAOA")]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for (method, n), g in q.groupby(["method", "assets"]):
        style = "o-" if "XY" in method else "s--"
        axes[0].plot(g["layers"], g["p_optimal"], style, label=f"{method}, n={n}")
        axes[1].plot(g["layers"], g["p_feasible"], style, label=f"{method}, n={n}")
    for n in sorted(q.assets.unique()):
        from math import comb
        axes[0].axhline(1 / comb(n, n // 2), color="gray", lw=0.6, ls=":")
    axes[0].set_ylabel("P(optimal portfolio)")
    axes[1].set_ylabel("P(feasible portfolio)")
    for ax in axes:
        ax.set_xlabel("QAOA depth p")
    axes[1].legend(fontsize=7)
    fig.suptitle("Portfolio selection with QAOA (gray: random feasible guess)")
    fig.tight_layout()
    fig.savefig(RESULTS / "portfolio_qaoa.png", dpi=200)


if __name__ == "__main__":
    main()
