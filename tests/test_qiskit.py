import unittest
import pytest
import numpy as np
from ptetools.qiskit import counts2dense


class TestQiskit(unittest.TestCase):
    def test_counts2dense(self):
        np.testing.assert_array_equal(counts2dense({'1': 100}, number_of_bits=1), np.array([  0, 100]))
        np.testing.assert_array_equal(counts2dense({'1': 100}, number_of_bits=2),  np.array([  0, 100,   0,   0]))



if __name__ == "__main__":
    unittest.main()
