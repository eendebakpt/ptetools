import logging
import pathlib
import random
import tempfile
from collections.abc import Sequence
from functools import lru_cache
from typing import Any, overload

import matplotlib.pyplot as plt
import numpy as np
import qiskit
import qiskit.quantum_info as qi
import qiskit.result
import qiskit_experiments.framework.containers.figure_data
from qiskit.circuit import Delay
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit_experiments.library.randomized_benchmarking.clifford_utils import CliffordUtils

from ptetools.tools import sorted_dictionary

CountsType = dict[str, int | float]
FractionsType = dict[str, float]
IntArray = np.typing.NDArray[int]
FloatArray = np.typing.NDArray[np.float64]
ComplexArray = np.typing.NDArray[np.complex128]


@overload
def counts2fractions(counts: Sequence[CountsType]) -> list[FractionsType]:
    ...


@overload
def counts2fractions(counts: CountsType) -> FractionsType:
    ...


def largest_remainder_rounding(fractions: FloatArray, total: int) -> IntArray:
    """Largest remainder rounding algorithm

        This function take a list of fractions and rounds to integers such that the sum adds
        up to total and the ratios are preserved.

        Notice: the algorithm we are using here is 'Largest Remainder'

    Code derived from https://stackoverflow.com/q/25271388
    """
    fractions = np.asarray(fractions)
    unround_numbers = (fractions / fractions.sum()) * total
    decimal_part_with_index = sorted(
        [(index, unround_numbers[index] % 1) for index in range(len(unround_numbers))], key=lambda y: y[1], reverse=True
    )
    remainder = total - sum(unround_numbers.astype(int))
    index = 0
    while remainder > 0:
        unround_numbers[decimal_part_with_index[index][0]] += 1
        remainder -= 1
        index = (index + 1) % fractions.size
    return [int(x) for x in unround_numbers]


def fractions2counts(f: list[CountsType] | CountsType, number_of_shots: int) -> list[CountsType] | CountsType:
    def f2c(x, number_of_shots: int):
        counts = largest_remainder_rounding(np.fromiter(x.values(), float), number_of_shots)
        return dict(zip(x.keys(), counts))

    if isinstance(f, dict):
        return f2c(f, number_of_shots)
    return [f2c(x, number_of_shots) for x in f]


if __name__ == "__main__":
    number_set = np.array([20.2, 20.2, 20.2, 20.2, 19.2]) / 100
    r = largest_remainder_rounding(number_set, 100)
    np.testing.assert_array_equal(r, [21, 20, 20, 20, 19])
    print(r, sum(r))

    fractions = dict(zip(range(3), [10.1, 80.4, 9.6]))
    assert fractions2counts(fractions, 100) == {0: 10, 1: 80, 2: 10}
    assert fractions2counts(fractions, 1024) == {0: 103, 1: 823, 2: 98}


def counts2fractions(counts: CountsType | Sequence[CountsType]) -> FractionsType | list[FractionsType]:
    """Convert list of counts to list of fractions"""
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


def circuit2matrix(circuit: QuantumCircuit) -> ComplexArray:
    op = qi.Operator(circuit)
    return op.data


def random_clifford_circuit(number_of_qubits: int) -> tuple[QuantumCircuit, int]:
    """Generate a circuit with a single random Clifford gate"""
    state = qiskit.QuantumCircuit(number_of_qubits, 0)  #
    if number_of_qubits == 2:
        cl_index = random.randrange(11520)
        cl = CliffordUtils.clifford_2_qubit_circuit(cl_index)
        state = state.compose(cl, (0, 1))
    elif number_of_qubits == 1:
        cl_index = random.randrange(24)
        cl = CliffordUtils.clifford_1_qubit_circuit(cl_index)
        state = state.compose(cl, (0,))
    else:
        raise NotImplementedError(f"number_of_qubits {number_of_qubits}")
    return state, cl_index


# %%


class RemoveGateByName(TransformationPass):  # type: ignore
    """Return a circuit with all gates with specified name removed.

    This transformation is not semantics preserving.
    """

    def __init__(self, gate_name: str, *args: Any, **kwargs: Any):
        """Remove all gates with specified name from a DAG

        Args:
            gate_name: Name of the gate to be removed from a DAG
        """
        super().__init__(*args, **kwargs)
        self._gate_name = gate_name

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the RemoveGateByName pass on `dag`."""

        dag.remove_all_ops_named(self._gate_name)

        return dag

    def __repr__(self) -> str:
        name = self.__class__.__module__ + "." + self.__class__.__name__
        return f"<{name} at 0x{id(self):x}: gate {self._gate_name}"


class RemoveZeroDelayGate(TransformationPass):  # type: ignore
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
        """Run the RemoveZeroDelayGate pass on `dag`."""

        for node in dag.op_nodes():
            if isinstance(node.op, Delay):
                if node.op.params[0] == 0:
                    dag.substitute_node_with_dag(node, self._empty_dag1)
        return dag

    def __repr__(self) -> str:
        name = self.__class__.__module__ + "." + self.__class__.__name__
        return f"<{name} at 0x{id(self):x}: gate {self._gate_name}"


if __name__ == "__main__":
    from qiskit.transpiler import PassManager

    qc = QuantumCircuit(2)
    qc.delay(0, 0)
    qc.barrier()
    qc.delay(0, 1)
    qc.draw()

    passes = [RemoveZeroDelayGate()]
    pm = PassManager(passes)
    r = pm.run([qc])
    print(r[0].draw())


def qiskit_experiments_to_figure(
    figure_data: qiskit_experiments.framework.containers.figure_data.FigureData,
    fig: int,
):
    """Convert qiskit experiment result to matplotlib figure window"""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_file_path = pathlib.Path(temp_dir, "fig.png")

        figure_data.figure.savefig(temp_file_path)
        im = plt.imread(temp_file_path)
        plt.figure(fig)
        plt.clf()
        plt.imshow(im)
        plt.axis("off")


@lru_cache
def delay_gate(duration: float, dt: float, round_dt: bool) -> float:
    n = duration / dt
    if round_dt:
        n = round(n)

    return Delay(n, unit="dt")


class ModifyDelayGate(TransformationPass):  # type: ignore
    """Return a circuit with small rotation gates removed."""

    def __init__(self, dt: float = 20e-9, round: bool = True) -> None:
        """Change delay gates to specified time step

        Args:
            epsilon: Threshold for rotation angle to be removed
            modulo2pi: If True, then rotations multiples of 2pi are removed as well
        """
        super().__init__()

        self.round = round
        self.dt = dt

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on `dag`.
        Args:
            dag: input dag.
        Returns:
            Output dag with Delay gates modified
        """
        for node in dag.op_nodes():
            if isinstance(node.op, Delay):
                params = node.op.params
                if node.op.unit == "s":
                    logging.info(f"{self.__class__.__name__}: found node with params {params}")
                    op = delay_gate(params[0], self.dt, self.round)
                    dag.substitute_node(node, op, inplace=True)
        return dag


if __name__ == "__main__":
    qc = QuantumCircuit(1)
    qc.delay(duration=123e-9, unit="s")
    p = ModifyDelayGate(dt=20e-9, round=True)
    qc = p(qc)
    print(qc.draw())
