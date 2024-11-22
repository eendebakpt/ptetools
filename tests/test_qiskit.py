import unittest

import numpy as np
from qiskit.circuit import QuantumCircuit

from ptetools.qiskit import (
    RemoveGateByName,
    RemoveZeroDelayGate,
    circuit2matrix,
    counts2dense,
    counts2fractions,
    fractions2counts,
    largest_remainder_rounding,
    random_clifford_circuit,
)


def circuit_instruction_names(qc):
    return [i.operation.name for i in qc]


class TestQiskit(unittest.TestCase):
    def test_counts2dense(self):
        np.testing.assert_array_equal(counts2dense({"1": 100}, number_of_bits=1), np.array([0, 100]))
        np.testing.assert_array_equal(counts2dense({"1": 100}, number_of_bits=2), np.array([0, 100, 0, 0]))

    def test_counts2fractions(self):
        assert counts2fractions({"1": 0}) == {"1": 0.0}
        assert counts2fractions({"1": 100, "0": 50}) == {"0": 0.3333333333333333, "1": 0.6666666666666666}

    def test_random_clifford_circuit(self):
        c, index = random_clifford_circuit(1)
        assert c.num_qubits == 1
        c, index = random_clifford_circuit(2)
        assert c.num_qubits == 2

    def test_RemoveGateByName(self):
        qc = QuantumCircuit(3)
        qc.h(0)
        qc.x(1)

        qc_transpiled = RemoveGateByName("none")(qc)
        self.assertEqual(circuit_instruction_names(qc_transpiled), circuit_instruction_names(qc))

        for name in ["x", "h", "dummy"]:
            qc_transpiled = RemoveGateByName(name)(qc)
            self.assertNotIn(name, circuit_instruction_names(qc_transpiled))

    def test_RemoveZeroDelayGate(self):
        qc = QuantumCircuit(3)
        qc.delay(0)
        qc.barrier()
        qc.delay(10, 0)
        qc.barrier()
        qc.delay(10)

        qc_transpiled = RemoveZeroDelayGate()(qc)
        self.assertEqual(
            circuit_instruction_names(qc_transpiled), ["barrier", "delay", "barrier", "delay", "delay", "delay"]
        )

    def test_fractions2counts(self):
        number_set = np.array([20.2, 20.2, 20.2, 20.2, 19.2]) / 100
        r = largest_remainder_rounding(number_set, 100)
        np.testing.assert_array_equal(r, [21, 20, 20, 20, 19])

        fractions = dict(zip(range(3), [10.1, 80.4, 9.6]))
        assert fractions2counts(fractions, 100) == {0: 10, 1: 80, 2: 10}
        assert fractions2counts(fractions, 1024) == {0: 103, 1: 823, 2: 98}

    def test_circuit2matrix(self):
        for k in range(1, 4):
            x = circuit2matrix(QuantumCircuit(k))
            np.testing.assert_array_equal(x, np.eye(2**k, dtype=complex))

        c = QuantumCircuit(1)
        c.x(0)
        x = circuit2matrix(c)
        expected = np.array([[0.0 + 0.0j, 1.0 + 0.0j], [1.0 + 0.0j, 0.0 + 0.0j]])
        np.testing.assert_array_equal(x, expected)


if __name__ == "__main__":
    unittest.main()
