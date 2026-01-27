import unittest
import warnings

import numpy as np

from ptetools._qtt import (
    angle_mean,
    decompose_projective_transformation,
    dehom,
    hom,
    mean_of_directions,
    pg_affine_to_homogeneous,
    pg_rotation2homogeneous,
    pg_rotx,
    pg_rotz,
    pg_scaling,
    pg_transl2homogeneous,
    projective_transformation,
    static_var,
)


# %%
class TestGeometryOperations(unittest.TestCase):
    def test_hom(self):
        pts = np.array([[1, 0], [1, 1], [2, 2]]).T
        expected = np.array([[1, 1, 2], [0, 1, 2], [1, 1, 1]])
        np.testing.assert_array_almost_equal(hom(pts), expected)

    def test_projective_transformation(self):
        warnings.filterwarnings("ignore", category=RuntimeWarning)
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

    def test_angle_mean(self):
        angles = np.array([0.0, 0.1, -0.1])
        mean = angle_mean(angles)
        self.assertAlmostEqual(mean, 0.0, places=5)

        angles = np.array([0.0, np.pi / 2])
        mean = angle_mean(angles)
        self.assertAlmostEqual(mean, np.pi / 4, places=5)

        angles = np.array([0.0, 0.0, 0.0])
        mean = angle_mean(angles)
        self.assertAlmostEqual(mean, 0.0, places=5)

        angles = np.array([0.0, np.pi, np.pi])
        weights = np.array([1.0, 1.0, 1.0])
        mean = angle_mean(angles, weights)
        self.assertAlmostEqual(np.abs(mean), np.pi, places=1)

    def test_dehom(self):
        pts_hom = np.array([[2, 4, 6], [4, 8, 12], [2, 2, 2]])
        expected = np.array([[1, 2, 3], [2, 4, 6]])
        np.testing.assert_array_almost_equal(dehom(pts_hom), expected)

        pts_hom = np.array([[1, 0], [0, 1], [1, 1]])
        expected = np.array([[1, 0], [0, 1]])
        np.testing.assert_array_almost_equal(dehom(pts_hom), expected)

    def test_pg_affine_to_homogeneous(self):
        affine = np.array([[2.0]])
        H = pg_affine_to_homogeneous(affine)
        expected = np.array([[2.0, 0.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(H, expected)

        affine = np.array([[1, 0], [0, 2]])
        H = pg_affine_to_homogeneous(affine)
        expected = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1]])
        np.testing.assert_array_almost_equal(H, expected)

        R = pg_rotx(0.5)
        H = pg_affine_to_homogeneous(R)
        np.testing.assert_array_almost_equal(H[:3, :3], R)
        np.testing.assert_array_almost_equal(H[3, :], [0, 0, 0, 1])

    def test_pg_transl2homogeneous(self):
        tr = [1, 2]
        H = pg_transl2homogeneous(tr)
        expected = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 2.0], [0.0, 0.0, 1.0]])
        np.testing.assert_array_almost_equal(H, expected)

        tr = [3, 4, 5]
        H = pg_transl2homogeneous(tr)
        self.assertEqual(H.shape, (4, 4))
        np.testing.assert_array_almost_equal(H[:3, 3], [3, 4, 5])

    def test_static_var(self):
        @static_var("counter", 0)
        def increment():
            increment.counter += 1
            return increment.counter

        self.assertEqual(increment.counter, 0)
        self.assertEqual(increment(), 1)
        self.assertEqual(increment(), 2)
        self.assertEqual(increment.counter, 2)
