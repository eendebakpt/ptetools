import functools
import operator

import qutip as qt

_paulis = {"X": qt.sigmax(), "Y": qt.sigmay(), "Z": qt.sigmaz(), "I": qt.qeye(2)}


def pauli_string_to_operator(s: str, scale: float = 1.0) -> qt.Qobj:
    """Convert Pauli string to qutip operator"""
    return functools.reduce(operator.and_, [_paulis[c] for c in s])


def basis2qutip(i: int, number_of_qubits: int) -> qt.Qobj:
    """Create ith basis vector for n-qubit state space"""
    w = qt.basis(2**number_of_qubits, i)
    w.dims = [[2] * number_of_qubits, [1] * number_of_qubits]
    return w
