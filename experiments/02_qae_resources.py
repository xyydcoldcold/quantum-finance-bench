"""Module A: gate-level resources of canonical QAE vs precision.

Transpiles the real circuits (no dense unitaries) to {CX, RZ, SX, X} and records qubits,
CNOT count, and depth as the number of evaluation qubits m and price qubits n grow.

    python experiments/02_qae_resources.py --max-eval-qubits 5
"""
import argparse

import matplotlib.pyplot as plt
import pandas as pd
from common import OPTION, RESULTS

from qfb.finance.black_scholes import lognormal_grid
from qfb.qae import build_grover, build_state_prep, canonical_qae_circuit, circuit_resources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-qubits", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--max-eval-qubits", type=int, default=5)
    args = ap.parse_args()

    rows = []
    for n in args.price_qubits:
        model = lognormal_grid(**OPTION, num_qubits=n)
        A = build_state_prep(model)
        Q = build_grover(A)
        rows.append(dict(price_qubits=n, eval_qubits=0, circuit="A", **circuit_resources(A)))
        rows.append(dict(price_qubits=n, eval_qubits=0, circuit="Q", **circuit_resources(Q)))
        for m in range(1, args.max_eval_qubits + 1):
            qc = canonical_qae_circuit(A, Q, m, use_unitary=False)
            res = circuit_resources(qc)
            rows.append(dict(price_qubits=n, eval_qubits=m, circuit="QAE", **res))
            print(f"n={n} m={m}: {res}")
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "qae_resources.csv", index=False)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for n, g in df[df.circuit == "QAE"].groupby("price_qubits"):
        ax.semilogy(g["eval_qubits"], g["cx"], "o-", label=f"{2**n}-point grid ({n} price qubits)")
    ax.set_xlabel("evaluation qubits m (precision ~ 1/2^m)")
    ax.set_ylabel("CNOT count after transpilation")
    ax.set_title("Canonical QAE circuit cost")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS / "qae_resources.png", dpi=200)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
