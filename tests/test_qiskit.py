import unittest

import numpy as np
from qiskit.circuit import QuantumCircuit

from ptetools.qiskit import RemoveGateByName, RemoveZeroDelayGate, counts2dense


def circuit_instruction_names(qc):
    return [i.operation.name for i in qc]


class TestQiskit(unittest.TestCase):
    def test_counts2dense(self):
        np.testing.assert_array_equal(counts2dense({"1": 100}, number_of_bits=1), np.array([0, 100]))
        np.testing.assert_array_equal(counts2dense({"1": 100}, number_of_bits=2), np.array([0, 100, 0, 0]))

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


if __name__ == "__main__":
    unittest.main()
