import unittest

import numpy as np

from ptetools._qtt import (
    decompose_projective_transformation,
    hom,
    mean_of_directions,
    pg_rotation2homogeneous,
    pg_rotx,
    pg_rotz,
    pg_scaling,
    pg_transl2homogeneous,
    projective_transformation,
)


class TestGeometryOperations(unittest.TestCase):
    def test_hom(self):
        pts = np.array([[1, 0], [1, 1], [2, 2]]).T
        expected = np.array([[1, 1, 2], [0, 1, 2], [1, 1, 1]])
        np.testing.assert_array_almost_equal(hom(pts), expected)

    def test_projective_transformation(self):
        x = np.array([[1.0, 0], [0, 2]])

        H = pg_transl2homogeneous([1.0, -1])
        y = projective_transformation(H, x)
        expected = np.array([[2.0, 1.0], [-1.0, 1.0]])
        np.testing.assert_array_almost_equal(y, expected)

        y = projective_transformation(np.eye(3), x)
        expected = np.array([[1.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_almost_equal(y, expected)

        with self.assertRaises(Exception):
            projective_transformation(np.eye(2), x)

        H = np.eye(3)
        H[2, 0] = -1
        y = projective_transformation(H, x)
        expected = np.array([[0.0, 0.0], [0.0, 2.0]])
        np.testing.assert_array_almost_equal(y[:, 1:], expected[:, 1:])

        x = np.array([[], []])
        y = projective_transformation(np.eye(3), x)
        np.testing.assert_array_almost_equal(x, y)

    def test_pg_rotx(self):
        identity = pg_rotx(90).dot(pg_rotx(-90))
        np.testing.assert_almost_equal(identity, np.eye(3))

        for phi in [0, 0.1, np.pi, 4]:
            rx = pg_rotx(phi)
            self.assertAlmostEqual(rx[1, 1], np.cos(phi))
            self.assertAlmostEqual(rx[2, 1], np.sin(phi))

    def test_pg_rotz(self):
        identity = pg_rotz(90).dot(pg_rotz(-90))
        np.testing.assert_almost_equal(identity, np.eye(3))

        for phi in [0, 0.1, np.pi, 4]:
            rz = pg_rotz(phi)
            self.assertAlmostEqual(rz[0, 0], np.cos(phi))
            self.assertAlmostEqual(rz[1, 0], np.sin(phi))

    def test_pg_scaling(self):
        H = pg_scaling([1, 2])
        np.testing.assert_array_equal(H, np.diag([1, 2, 1]))
        H = pg_scaling([2], [1])
        np.testing.assert_array_equal(H, np.array([[2, -1], [0, 1.0]]))
        with self.assertRaises(ValueError):
            pg_scaling([1], [1, 2])

    def test_pg_rotation2homogeneous(self):
        R = pg_rotx(0.12)
        H = pg_rotation2homogeneous(R)
        np.testing.assert_almost_equal(R, H[:3, :3])
        self.assertIsNotNone(H)

    def test_decompose_projective_transformation(self):
        R = pg_rotation2homogeneous(pg_rotx(np.pi / 2))
        affine, scaling, projective, _ = decompose_projective_transformation(R)
        self.assertIsInstance(affine, np.ndarray)
        np.testing.assert_array_almost_equal(scaling @ affine @ projective, R)

        R = pg_rotation2homogeneous(pg_rotx(0.012))
        affine, scaling, projective, _ = decompose_projective_transformation(R)
        np.testing.assert_array_almost_equal(scaling @ affine @ projective, R)

    def test_mean_of_directions(self):
        directions = [[1, 0], [1, 0.1], [1, -0.1]]
        angle = mean_of_directions(directions)
        angle = np.mod(angle, np.pi)
        self.assertAlmostEqual(angle, np.pi / 2)

        directions = [[1, 0], [-1, 0]]
        angle = mean_of_directions(directions)
        angle = np.mod(angle, np.pi)
        self.assertAlmostEqual(angle, np.pi / 2)
