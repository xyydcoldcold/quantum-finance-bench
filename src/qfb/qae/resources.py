"""Gate-level resource counts for the QAE circuits (no dense unitaries)."""
from __future__ import annotations

from qiskit import QuantumCircuit, transpile

BASIS = ["cx", "rz", "sx", "x"]


def circuit_resources(qc: QuantumCircuit, optimization_level: int = 1) -> dict:
    """Transpile to a CNOT + single-qubit basis and report qubits, CNOT count, depth."""
    t = transpile(qc, basis_gates=BASIS, optimization_level=optimization_level)
    ops = t.count_ops()
    return {
        "qubits": t.num_qubits,
        "cx": int(ops.get("cx", 0)),
        "single_qubit": int(sum(v for k, v in ops.items() if k in ("rz", "sx", "x"))),
        "depth": int(t.depth()),
        "two_qubit_depth": int(t.depth(lambda inst: inst.operation.num_qubits == 2)),
    }
