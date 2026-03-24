import itertools
import logging
import math
import pathlib
import random
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from functools import lru_cache
from numbers import Real
from typing import Any, overload

import matplotlib.pyplot as plt
import numpy as np
import qiskit
import qiskit.circuit
import qiskit.circuit.operation
import qiskit.converters
import qiskit.quantum_info as qi
import qiskit.result
import qiskit_experiments.framework.containers.figure_data
import qutip.core.superop_reps
from qiskit.circuit import Delay, Instruction
from qiskit.circuit.library import (
    CRXGate,
    CRYGate,
    CRZGate,
    PhaseGate,
    RXGate,
    RYGate,
    RZGate,
    U1Gate,
    U2Gate,
    U3Gate,
    UGate,
)
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit.dagcircuit import DAGCircuit
from qiskit.transpiler.basepasses import TransformationPass
from qiskit_experiments.library.randomized_benchmarking.clifford_utils import CliffordUtils
from qutip import Qobj

CountsType = Mapping[str, int | float]
FractionsType = Mapping[str, float]
IntArray = np.typing.NDArray[np.int64 | np.int32]
IntArrayLike = np.typing.NDArray[np.int64 | np.int32] | list[int] | tuple[int, ...]
FloatArray = np.typing.NDArray[np.float64]
ComplexArray = np.typing.NDArray[np.complex128]


# %% Bit conversions


def generate_bitstring_tuples(number_of_bits: int) -> Iterator[tuple[int, ...]]:
    return itertools.product(*((0, 1),) * (number_of_bits))


def generate_bitstrings(number_of_bits: int) -> list[str]:
    """Generate bitstrings for specified number of bits

    Example:
        >>> generate_bitstrings(2)
        ['00', '01', '10', '11']
    """
    fmt = f"{{:0{number_of_bits}b}}"
    return [fmt.format(w) for w in range(2**number_of_bits)]


def invert_permutation(permutation) -> IntArray:
    inv = np.empty_like(permutation)
    inv[permutation] = np.arange(len(inv), dtype=int)
    return inv


def bitlist_to_int(bitlist: Sequence[int]) -> int:
    out = 0
    for bit in bitlist:
        out = (out << 1) | bit
    return out


def index2bitstring(index: int, number_of_bits: int) -> str:
    fmt = f"{{:0{number_of_bits}b}}"
    return fmt.format(index)


def permute_bits(idx: int, permutation: Sequence[int]) -> int:
    """Permute position of bits in an integer"""
    number_of_bits = len(permutation)
    bs_reversed = index2bitstring(idx, number_of_bits)[::-1]
    permuted_bs = [int(bs_reversed[v]) for v in permutation][::-1]
    pidx = bitlist_to_int(permuted_bs)
    return pidx


def permute_string(string: str, permutation: Sequence[int]) -> str:
    """Permute string characters"""
    permuted = [string[pidx] for pidx in permutation]
    return "".join(permuted)


def permute_counts(counts: CountsType, permutation: Sequence[int]) -> CountsType:
    """Permute bits in a counts for fractions object

    For the bits we use the Qiskit convention: LSB has index zero
    """
    return {permute_string(bitstring[::-1], permutation)[::-1]: value for bitstring, value in counts.items()}


def generate_state_labels(k: int, latex: bool = True):
    """Generate state labels for the specified number of qubits"""
    if latex:
        return [f"$|{b}\\rangle$" for b in generate_bitstrings(k)]
    else:
        return [f"|{b}>" for b in generate_bitstrings(k)]


if __name__ == "__main__":  # pragma: no cover
    permutation = [0, 1, 3, 2]
    print(permute_bits(idx=1, permutation=permutation))

    permutation = [0, 1, 3, 2]
    assert permute_bits(idx=1, permutation=permutation) == 1

    assert permute_bits(idx=0, permutation=[1, 0]) == 0
    assert permute_bits(idx=1, permutation=[1, 0]) == 2
    assert permute_bits(idx=1, permutation=[1, 2, 0]) == 4
    assert permute_bits(idx=3, permutation=[3, 4, 0, 1, 2]) == 12

    assert permute_counts({"00": 10, "01": 20}, [1, 0]) == {"00": 10, "10": 20}

    np.testing.assert_array_equal(invert_permutation([0, 1, 3, 2]), np.array([0, 1, 3, 2]))

    counts: CountsType = {"1110": 945, "0010": 7, "1011": 16}
    permutation = [1, 0, 2, 3]
    assert permute_counts(counts, permutation) == {"1101": 945, "0001": 7, "1011": 16}

    generate_state_labels(3)

# %%


def largest_remainder_rounding(fractions: FloatArray, total: int) -> list[int]:
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


def fractions2counts(
    f: list[CountsType] | CountsType, number_of_shots: int, integer_rounding: bool = True
) -> list[CountsType] | CountsType:
    if integer_rounding is True:

        def f2c(x, number_of_shots: int):
            counts = largest_remainder_rounding(np.fromiter(x.values(), float), number_of_shots)
            return dict(zip(x.keys(), counts))
    else:

        def f2c(x, number_of_shots: int):
            counts = {key: number_of_shots * value for key, value in x.items()}
            return counts

    if isinstance(f, dict):
        return f2c(f, number_of_shots)
    return [f2c(x, number_of_shots) for x in f]


if __name__ == "__main__":  # pragma: no cover
    number_set = np.array([20.2, 20.2, 20.2, 20.2, 19.2]) / 100
    r = largest_remainder_rounding(number_set, 100)
    np.testing.assert_array_equal(r, [21, 20, 20, 20, 19])
    print(r, sum(r))

    fractions = dict(zip(range(3), [10.1, 80.4, 9.6]))
    assert fractions2counts(fractions, 100) == {0: 10, 1: 80, 2: 10}
    assert fractions2counts(fractions, 1024) == {0: 103, 1: 823, 2: 98}


@overload
def counts2fractions(counts: CountsType) -> FractionsType: ...


@overload
def counts2fractions(counts: FractionsType) -> FractionsType: ...


@overload
def counts2fractions(counts: Sequence[CountsType | FractionsType]) -> list[FractionsType]: ...


def counts2fractions(
    counts: CountsType | FractionsType | Sequence[CountsType | FractionsType],
) -> FractionsType | list[FractionsType]:
    """Convert list of counts to list of fractions"""
    if isinstance(counts, Sequence):
        return [counts2fractions(c) for c in counts]  # ty: ignore
    total = sum(counts.values())
    total = total or 1  # corner case with no selected shots

    return {k: float(counts[k] / total) for k in sorted(counts)}


def normalize_probability(probabilities: FloatArray) -> FloatArray:
    """Normalize probabilities to have sum 1 and in interval [0, 1]"""
    w = np.minimum(np.maximum(probabilities, 0.0), 1.0)
    w = w / np.sum(w)
    return w


def counts2dense(c: CountsType, number_of_bits: int) -> np.ndarray:
    """Convert dictionary with fractions or counts to a dense array"""
    d = np.zeros(2**number_of_bits, dtype=np.array(sum(c.values())).dtype)
    for k, v in c.items():
        idx = int(k.replace(" ", ""), base=2)
        d[idx] = v
    return d


def dense2sparse(d: IntArray) -> CountsType:
    """Convert a dense array to a sparse counts dictionary"""
    d = np.asanyarray(d)
    number_of_bits = int(np.log2(d.size))
    fmt = f"{{:0{number_of_bits}b}}"
    bb = [fmt.format(idx) for idx in range(2**number_of_bits)]
    counts = {bitstring: d[idx].item() for idx, bitstring in enumerate(bb)}
    counts = {key: value for key, value in counts.items() if value}
    return counts


def normalize_fractions(f: FloatArray) -> FloatArray:
    """Normalize fractions by clipping to [0, 1] range and scale to norm 1"""
    f = np.clip(f, 0, 1)
    return f / sum(f)


def circuit2matrix(circuit: QuantumCircuit, decimals: int | None = 5) -> ComplexArray:
    """Deprecated: use circuit_to_matrix instead"""
    return circuit_to_matrix(circuit, decimals)


def circuit_to_matrix(circuit: QuantumCircuit, decimals: int | None = 5) -> ComplexArray:
    op = qi.Operator(circuit)
    U = op.data
    if decimals is not None:
        U = np.real_if_close(U)
        U = np.round(U, decimals=decimals)
    return U


def random_clifford_circuit(number_of_qubits: int) -> tuple[QuantumCircuit, int]:
    """Generate a circuit with a single random Clifford gate"""
    state: QuantumCircuit = qiskit.QuantumCircuit(number_of_qubits, 0)
    if number_of_qubits == 2:
        cl_index = random.randrange(11520)
        cl = CliffordUtils.clifford_2_qubit_circuit(cl_index)
        state.compose(cl, (0, 1), inplace=True)
    elif number_of_qubits == 1:
        cl_index = random.randrange(24)
        cl = CliffordUtils.clifford_1_qubit_circuit(cl_index)
        state.compose(cl, (0,), inplace=True)
    else:
        raise NotImplementedError(f"number_of_qubits {number_of_qubits}")
    return state, cl_index


# %%


class RemoveGateByName(TransformationPass):
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

    def run(self, dag: DAGCircuit) -> DAGCircuit:  # qiskit upstream issue
        """Run the RemoveGateByName pass on `dag`."""

        dag.remove_all_ops_named(self._gate_name)

        return dag

    def __repr__(self) -> str:
        name = self.__class__.__module__ + "." + self.__class__.__name__
        return f"<{name} at 0x{id(self):x}: gate {self._gate_name}"


class RemoveZeroDelayGate(TransformationPass):
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

    def run(self, dag: DAGCircuit) -> DAGCircuit:  # qiskit upstream issue
        """Run the RemoveZeroDelayGate pass on `dag`."""

        for node in dag.op_nodes():
            if isinstance(node.op, Delay):
                if node.op.params[0] == 0:
                    dag.substitute_node_with_dag(node, self._empty_dag1)
        return dag


class ReplaceGate(TransformationPass):
    def __init__(
        self,
        gate: type[qiskit.circuit.Gate],
        replacement_circuit: QuantumCircuit,
        qubits=None,
    ):
        """Replace selected gate type optionally on selected qubits.

        Args:
            gate: gate type to match (e.g. RXGate).
            replacement_circuit: circuit used to replace each matching instance.
            qubits: None to replace all matching gates, or int/list of qubit indices to limit replacement.
        """
        super().__init__()
        self.gate = gate
        self.replacement_circuit = replacement_circuit
        self.replacement_dag = circuit_to_dag(self.replacement_circuit)

        if qubits is None:
            self.qubit_set = None
        elif isinstance(qubits, int):
            self.qubit_set = {qubits}
        else:
            self.qubit_set = set(qubits)

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the substitution pass on `dag`."""
        for node in dag.op_nodes(self.gate):
            if self.qubit_set is not None:
                node_qubits = {dag.qubits.index(q) for q in node.qargs}
                if not node_qubits <= self.qubit_set:
                    continue
            dag.substitute_node_with_dag(node, self.replacement_dag)
        return dag


if __name__ == "__main__":  # pragma: no cover
    from qiskit.transpiler import PassManager

    gate = qiskit.circuit.library.CXGate
    replacement_circuit = QuantumCircuit(2)
    replacement_circuit.barrier()
    replacement_circuit.cx(0, 1)
    replacement_circuit.barrier()

    qc = QuantumCircuit(2)
    qc.cx(0, 1)
    qpass = ReplaceGate(gate, replacement_circuit)
    d = qpass(qc)
    print(d.draw())

    qc = QuantumCircuit(2)
    qc.delay(0, 0)
    qc.barrier()
    qc.delay(0, 1)
    qc.draw()

    passes: list[TransformationPass] = [RemoveZeroDelayGate()]
    pm = PassManager(passes)  # type: ignore[arg-type]
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
def delay_gate(duration: float, dt: float, round_dt: bool) -> qiskit.circuit.operation.Operation:
    n = duration / dt
    if round_dt:
        n = round(n)

    return Delay(n, unit="dt")


class ModifyDelayGate(TransformationPass):
    """Return a circuit with small rotation gates removed."""

    def __init__(self, dt: float = 20e-9, round_dt: bool = True) -> None:
        """Change delay gates to specified time unit"""
        super().__init__()

        self.round_dt = round_dt
        self.dt = dt

    def run(self, dag: DAGCircuit) -> DAGCircuit:  # qiskit upstream issue
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
                    op = delay_gate(params[0], self.dt, self.round_dt)
                    dag.substitute_node(node, op, inplace=True)
        return dag


if __name__ == "__main__":  # pragma: no cover
    qc = QuantumCircuit(1)
    qc.delay(duration=123e-9, unit="s")
    p = ModifyDelayGate(dt=20e-9, round_dt=True)
    qc = p(qc)
    print(qc.draw())


def choi_to_unitary(choi: ComplexArray) -> ComplexArray:
    """Project choi matrix to closest unitary"""
    n = int(math.log2(choi.shape[0])) // 2
    bb = [[2] * n, [2] * n]
    b = [bb] * 2
    hermitian_choi = (choi + choi.conj().T) / 2  # enforce Hermiticity
    choi_qobj = Qobj(hermitian_choi, dims=b, superrep="choi")

    krauss = qutip.core.superop_reps.to_kraus(choi_qobj)
    dominant_idx = np.nanargmax([np.abs(np.linalg.det(c.full())) for c in krauss])
    U = krauss[dominant_idx].full()

    phase = np.exp(-np.angle(U[0, 0]) * 1j)
    U = phase * U
    return U


if __name__ == "__main__":  # pragma: no cover
    import qutip

    X = qutip.sigmax()
    Y = qutip.sigmay()
    Z = qutip.sigmaz()
    for U in [X, Y & Z]:
        s = qutip.core.superop_reps.to_super(U)
        choi_qobj = qutip.core.superop_reps.to_choi(s)
        choi = choi_qobj.full()
        Ur = choi_to_unitary(choi)
        IC = Ur @ U.full().conjugate().T
        IC = np.exp(-np.angle(IC[0, 0]) * 1j) * IC
        np.testing.assert_almost_equal(IC, np.eye(IC.shape[0]))

# %% Vendored from qtt


class RemoveSmallRotations(TransformationPass):
    """Return a circuit with small rotation gates removed."""

    def __init__(self, epsilon: float = 0, modulo2pi: bool = False) -> None:
        """Remove all small rotations from a circuit

        Args:
            epsilon: Threshold for rotation angle to be removed
            modulo2pi: If True, then rotations multiples of 2pi are removed as well
        """
        super().__init__()

        self.epsilon = epsilon
        self._empty_dag1 = circuit_to_dag(QuantumCircuit(1), copy_operations=False)
        self._empty_dag2 = circuit_to_dag(QuantumCircuit(2), copy_operations=False)
        self.mod2pi = modulo2pi

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the pass on `dag`.
        Args:
            dag: input dag.
        Returns:
            Output dag with small rotations removed
        """

        def modulo_2pi(x: float) -> float:
            x = float(x)
            return float(np.mod(x + np.pi, 2 * np.pi) - np.pi)

        for node in dag.op_nodes():
            if isinstance(node.op, (PhaseGate, RXGate, RYGate, RZGate)):
                if node.op.is_parameterized():
                    # for parameterized gates we do not optimize
                    pass
                else:
                    phi = float(node.op.params[0])
                    if self.mod2pi:
                        phi = modulo_2pi(phi)
                    if np.abs(phi) <= self.epsilon:
                        dag.substitute_node_with_dag(node, self._empty_dag1)
            elif isinstance(node.op, (CRXGate, CRYGate, CRZGate)):
                if node.op.is_parameterized():
                    # for parameterized gates we do not optimize
                    pass
                else:
                    phi = float(node.op.params[0])
                    if self.mod2pi:
                        phi = modulo_2pi(phi)
                    if np.abs(phi) <= self.epsilon:
                        dag.substitute_node_with_dag(node, self._empty_dag2)
        return dag


def _is_numeric_parameter(value: Any) -> bool:
    return isinstance(value, (Real, np.number))


def _u2_gate(qc: QuantumCircuit, phi: Any, lam: Any) -> None:
    """Add decomposition of U2 gate to quantum circuit"""
    if _is_numeric_parameter(phi) and _is_numeric_parameter(lam):
        phi_float = float(phi)
        lam_float = float(lam)
    else:
        phi_float = None
        lam_float = None

    if (
        phi_float is not None
        and lam_float is not None
        and np.isclose(phi_float, 0.0, atol=1e-12)
        and np.isclose(lam_float, 0.0, atol=1e-12)
    ):
        qc.ry(np.pi / 2, 0)
    elif (
        phi_float is not None
        and lam_float is not None
        and np.isclose(np.mod(phi_float + np.pi, 2 * np.pi) - np.pi, -np.pi / 2, atol=1e-12)
        and np.isclose(np.mod(lam_float + np.pi, 2 * np.pi) - np.pi, np.pi / 2, atol=1e-12)
    ):
        qc.rx(np.pi / 2, 0)
    else:
        qc.rz(lam - np.pi / 2, 0)
        qc.rx(np.pi / 2, 0)
        qc.rz(phi + np.pi / 2, 0)


class DecomposeU(TransformationPass):
    def __init__(self) -> None:
        """Decompose U gates into elementary rotations Rx(pi/2), Ry(pi/2), Rz

        The U gates are decomposed using McKay decomposition.
        """
        super().__init__()

    @staticmethod
    def _decompose_three_parameter_u(qc: QuantumCircuit, theta: Any, phi: Any, lam: Any) -> None:
        """Decompose U(theta, phi, lam) into elementary rotations."""
        if _is_numeric_parameter(theta) and _is_numeric_parameter(phi) and _is_numeric_parameter(lam):
            theta_float = float(theta)
            phi_float = float(phi)
            lam_float = float(lam)

            theta_mod = float(np.mod(theta_float, 2 * np.pi))
            phi_mod = float(np.mod(phi_float + np.pi, 2 * np.pi) - np.pi)
            lam_mod = float(np.mod(lam_float + np.pi, 2 * np.pi) - np.pi)

            if (
                np.isclose(theta_mod, np.pi, atol=1e-12)
                and np.isclose(phi_mod, -np.pi / 2, atol=1e-12)
                and np.isclose(lam_mod, np.pi / 2, atol=1e-12)
            ):
                qc.rx(np.pi / 2, 0)
                qc.rx(np.pi / 2, 0)
                return

            if np.isclose(theta_mod, np.pi / 2, atol=1e-12):
                _u2_gate(qc, phi_float, lam_float)
                return

            if np.isclose(phi_float, 0.0, atol=1e-12) and np.isclose(lam_float, 0.0, atol=1e-12):
                if np.isclose(theta_float, -np.pi / 2, atol=1e-12) or np.isclose(theta_mod, 3 * np.pi / 2, atol=1e-12):
                    qc.ry(-np.pi / 2, 0)
                    return
                if np.isclose(theta_float, np.pi, atol=1e-12) or np.isclose(theta_float, -np.pi, atol=1e-12):
                    qc.ry(np.pi / 2, 0)
                    qc.ry(np.pi / 2, 0)
                    return

            # from https://arxiv.org/pdf/1707.03429.pdf
            qc.rz(lam_float, 0)
            qc.rx(np.pi / 2, 0)
            qc.rz(theta_mod + np.pi, 0)
            qc.rx(np.pi / 2, 0)
            qc.rz(phi_float + np.pi, 0)
            return

        # from https://arxiv.org/pdf/1707.03429.pdf
        qc.rz(lam, 0)
        qc.rx(np.pi / 2, 0)
        qc.rz(theta + np.pi, 0)
        qc.rx(np.pi / 2, 0)
        qc.rz(phi + np.pi, 0)

    @staticmethod
    @lru_cache(maxsize=1000)
    def _ugate_replacement_circuit(parameters: tuple[Any, ...]) -> QuantumCircuit:
        """Return circuit used for replacement of the U gate"""
        qc = QuantumCircuit(1)

        match len(parameters):
            case 3:
                theta, phi, lam = parameters
                DecomposeU._decompose_three_parameter_u(qc, theta, phi, lam)
            case 2:
                _u2_gate(qc, *parameters)
            case 1:
                (lam,) = parameters
                qc.rz(lam, 0)
            case _:
                raise ValueError(f"length of parameters {parameters} invalid")
        return qc

    def ugate_replacement_circuit(self, ugate: Instruction) -> QuantumCircuit:
        """Return circuit used for replacement of the U gate"""
        if not isinstance(ugate, (U3Gate, UGate, U2Gate, U1Gate, PhaseGate)):
            raise TypeError(f"unsupported gate type {type(ugate).__name__}")

        return self._ugate_replacement_circuit(tuple(ugate.params))

    def run(self, dag: DAGCircuit) -> DAGCircuit:
        """Run the Decompose pass on `dag`.

        Args:
            dag: input DAG.

        Returns:
            Output DAG where ``U`` gates have been decomposed.
        """
        # Walk through the DAG and expand each node if required
        for node in dag.op_nodes():
            if isinstance(node.op, (PhaseGate, U1Gate, U2Gate, U3Gate, UGate)):
                subdag = circuit_to_dag(self.ugate_replacement_circuit(node.op), copy_operations=False)
                dag.substitute_node_with_dag(node, subdag)
        return dag
