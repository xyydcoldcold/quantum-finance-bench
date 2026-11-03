"""Graph view of the market: Max-Cut on a correlation-distance graph.

Edge weight d_ij = sqrt(2 (1 - rho_ij)) (Mantegna distance) is small for strongly
correlated assets. A maximum cut maximizes the distance crossing the cut, which is the
same as minimizing within-group distance: it splits assets into two internally
correlated groups (e.g. sectors), useful for diversification and hedging.

Max-Cut as a QUBO:  cut(x) = sum_{(i,j)} w_ij (x_i + x_j - 2 x_i x_j);  minimize -cut(x).
"""
from __future__ import annotations

import networkx as nx
import numpy as np

from .qubo import QUBO


def correlation_distance_graph(returns) -> nx.Graph:
    rho = np.corrcoef(np.asarray(returns), rowvar=False)
    n = rho.shape[0]
    names = list(getattr(returns, "columns", range(n)))
    g = nx.Graph()
    for i in range(n):
        g.add_node(i, name=names[i])
        for j in range(i + 1, n):
            g.add_edge(i, j, weight=float(np.sqrt(max(0.0, 2 * (1 - rho[i, j])))))
    return g


def maxcut_qubo(g: nx.Graph) -> QUBO:
    n = g.number_of_nodes()
    Q = np.zeros((n, n))
    c = np.zeros(n)
    for i, j, d in g.edges(data=True):
        w = d.get("weight", 1.0)
        c[i] -= w
        c[j] -= w
        Q[i, j] += w  # symmetric pair contributes 2*w*x_i*x_j in x^T Q x
        Q[j, i] += w
    return QUBO(Q=Q, c=c)


def cut_value(g: nx.Graph, x) -> float:
    return float(sum(d.get("weight", 1.0) for i, j, d in g.edges(data=True) if x[i] != x[j]))
