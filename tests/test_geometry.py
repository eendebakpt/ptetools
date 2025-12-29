import unittest

import numpy as np

from ptetools.geometry import angle_mean


class TestGeometryOperations(unittest.TestCase):
    def test_angle_mean(self):
        np.testing.assert_almost_equal(angle_mean([0.1, 0.2]), 0.15)

        np.testing.assert_almost_equal(angle_mean([0.0, 2 * np.pi]), 0.0)
        np.testing.assert_almost_equal(angle_mean([0.01, np.pi]), 1.575796326794898)

        np.testing.assert_almost_equal(angle_mean([0.0, 0.123], [0.0, 1.0]), 0.123)

        np.testing.assert_almost_equal(angle_mean([0.0, np.pi / 2], [0.5, 0.5]), np.pi / 4)
        np.testing.assert_almost_equal(angle_mean([0.0, np.pi / 2], [0.25, 0.75]), 3 * np.pi / 4)
