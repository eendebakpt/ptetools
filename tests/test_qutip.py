import unittest

import numpy as np
import qutip as qt

from ptetools.qutip import basis2qutip, pauli_string_to_operator


class TestQutip(unittest.TestCase):
    def test_pauli_string_to_operator(self):
        assert pauli_string_to_operator("I") == qt.qeye(2)
        assert pauli_string_to_operator("XY") == qt.sigmax() & qt.sigmay()

    def test_basis2qutip(self):
        assert basis2qutip(0, 1) == qt.basis(2, 0)
        b3_0 = basis2qutip(0, 3)
        assert b3_0.shape == (8, 1)
        np.testing.assert_array_equal(b3_0.data.as_ndarray(), np.array([[1.0, 0, 0, 0, 0, 0, 0, 0]]).T)


if __name__ == "__main__":
    unittest.main()
