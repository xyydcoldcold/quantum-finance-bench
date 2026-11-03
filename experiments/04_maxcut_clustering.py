"""Module B (graph theory): split assets into two correlation clusters with QAOA Max-Cut.

Builds the Mantegna correlation-distance graph, solves Max-Cut exactly, with networkx's
one-exchange local search, and with QAOA, and checks whether the cut recovers the sectors.

    python experiments/04_maxcut_clustering.py --assets 8 --layers 1 2 3
"""
import argparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from common import RESULTS
from networkx.algorithms.approximation import one_exchange

from qfb.data import returns_from_price_csv, synthetic_returns
from qfb.qaoa import brute_force, correlation_distance_graph, maxcut_qubo, run_qaoa
from qfb.qaoa.maxcut import cut_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", type=int, default=8)
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--prices-csv", default=None)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    if args.prices_csv:
        rets = returns_from_price_csv(args.prices_csv).iloc[:, : args.assets]
    else:
        rets = synthetic_returns(args.assets, num_sectors=2, seed=args.seed)
    g = correlation_distance_graph(rets)
    qubo = maxcut_qubo(g)
    x_opt, e_min, _, _ = brute_force(qubo)
    best_cut = -e_min
    _, (s1, _) = one_exchange(g, weight="weight", seed=args.seed)
    x_ls = np.array([1 if i in s1 else 0 for i in range(args.assets)])
    print(f"exact cut {best_cut:.4f} {x_opt} | one-exchange {cut_value(g, x_ls):.4f} {x_ls}")
    sectors = np.array([int(c.split('_S')[1]) if '_S' in str(c) else -1 for c in rets.columns])

    rows = [dict(method="Exact", layers=0, cut=best_cut, ratio=1.0, p_optimal=np.nan),
            dict(method="One-exchange", layers=0, cut=cut_value(g, x_ls), ratio=cut_value(g, x_ls) / best_cut,
                 p_optimal=np.nan)]
    for p in args.layers:
        r = run_qaoa(qubo, layers=p, steps=args.steps, seed=args.seed)
        exp_cut = -r.expected_energy
        rows.append(dict(method="QAOA", layers=p, cut=exp_cut, ratio=exp_cut / best_cut, p_optimal=r.p_optimal))
        print(f"p={p}: expected cut {exp_cut:.4f} (ratio {exp_cut / best_cut:.3f}), P(opt)={r.p_optimal:.3f}, "
              f"argmax {r.most_likely}")
    if (sectors >= 0).all():
        match = max(np.mean(x_opt == sectors), np.mean(x_opt != sectors))
        print(f"Optimal cut agrees with the true sector labels on {match:.0%} of assets")
    pd.DataFrame(rows).to_csv(RESULTS / "maxcut.csv", index=False)

    pos = nx.spring_layout(g, weight=None, seed=0)
    fig, ax = plt.subplots(figsize=(5, 5))
    nx.draw_networkx(g, pos, ax=ax, node_color=["tab:blue" if b else "tab:orange" for b in x_opt],
                     labels={i: str(rets.columns[i]) for i in g.nodes}, font_size=7,
                     width=[1.5 * (2 - d["weight"]) for _, _, d in g.edges(data=True)], edge_color="0.7")
    ax.set_title("Max-Cut partition of the correlation-distance graph")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(RESULTS / "maxcut.png", dpi=200)


if __name__ == "__main__":
    main()
