from collections.abc import Sequence
from typing import overload

import numpy as np

from ptetools.tools import sorted_dictionary

CountsType = dict[str, int | float]


@overload
def counts2fractions(counts: Sequence[CountsType]) -> list[CountsType]:
    ...


@overload
def counts2fractions(counts: CountsType) -> CountsType:
    ...


def counts2fractions(counts: CountsType | Sequence[CountsType]) -> CountsType | list[CountsType]:
    if isinstance(counts, Sequence):
        return [counts2fractions(c) for c in counts]
    total = sum(counts.values())
    if total == 0:
        # corner case with no selected shots
        total = 1

    return sorted_dictionary({k: v / total for k, v in counts.items()})


def counts2dense(c: CountsType, number_of_bits: int) -> np.ndarray:
    """Convert dictionary with fractions or counts to a dense array"""
    d = np.zeros(2**number_of_bits, dtype=np.array(sum(c.values())).dtype)
    for k, v in c.items():
        idx = int(k.replace(" ", ""), base=2)
        d[idx] = v
    return d


def dense2sparse(d: np.ndarray) -> CountsType:
    """Convert dictionary with fractions or counts to a dense array"""
    d = np.asanyarray(d)
    number_of_bits = int(np.log2(d.size))
    fmt = f"{{:0{number_of_bits}b}}"
    bb = [fmt.format(idx) for idx in range(2**number_of_bits)]
    counts = {bitstring: d[idx].item() for idx, bitstring in enumerate(bb)}
    return counts


if __name__ == "__main__":
    print(counts2dense({"1 0": 1.0}, 2))
    print(counts2fractions({"11": 20, "00": 30}))
    print(counts2fractions([{"11": 20, "00": 30}]))
    print(dense2sparse([2, 0, 4, 2]))

# %%


from typing import Any

import numpy as np
import qiskit
import qiskit.result
from qiskit.circuit import Delay
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass


class RemoveDelayGate(TransformationPass):  # type: ignore
    """Return a circuit with all zero duration delay gates removed.

    This transformation is not semantics preserving.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        """Remove all zero duration delay gates from a DAG

        Args:
            gate_name: Name of the gate to be removed from a DAG
        """
        self._empty_dag1 = qiskit.converters.circuit_to_dag(QuantumCircuit(1))
        super().__init__(*args, **kwargs)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the RemoveGateByName pass on `dag`."""

        dag.remove_all_ops_named("delay")

        for node in dag.op_nodes():
            if isinstance(node.op, Delay):
                if node.op.params[0] == 0:
                    dag.substitute_node_with_dag(node, self._empty_dag1)
        return dag

    def __repr__(self) -> str:
        name = self.__class__.__module__ + "." + self.__class__.__name__
        return f"<{name} at 0x{id(self):x}: gate {self._gate_name}"


if __name__ == "__main__":
    from qiskit import QuantumCircuit
    from qiskit.transpiler import PassManager

    qc = QuantumCircuit(2)
    qc.delay(0, 0)
    qc.barrier()
    qc.delay(0, 1)
    qc.draw()

    passes = [RemoveDelayGate()]
    pm = PassManager(passes)
    r = pm.run([qc])
    print(r[0].draw())
