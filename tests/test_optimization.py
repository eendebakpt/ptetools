import unittest

import lmfit
import matplotlib.pyplot as plt
import numpy as np
import pandas
import scipy.optimize
from numpy.random import default_rng

from ptetools.optimization import AverageDecreaseTermination, OptimizerCallback, OptimizerLog


class TestOptimizationUtilities(unittest.TestCase):
    def test_OptimizerLog(self):
        log = OptimizerLog([], [])
        log.plot()
        plt.close("all")

    def test_OptimizerCallback(self):
        rng = default_rng(123)

        def rosen(params, a=1, b=100, noise=0):
            """Rosenbrock function"""
            v = (a - params[0]) ** 2 + b * (params[1] - params[0] ** 2) ** 2
            v += noise * (rng.random() - 0.5)
            return v

        def objective(x):
            return rosen(x, 0.01, 1)

        oc = OptimizerCallback(show_progress=False)

        result = scipy.optimize.minimize(objective, [0.5, 0.9], callback=oc.scipy_callback)
        self.assertEqual(result.success, True)
        self.assertEqual(oc.number_of_evaluations(), result.nit)
        self.assertIsInstance(oc.data, pandas.DataFrame)

        oc.clear()

        def linear_function(x, a, b):
            return a * x + b

        lmfit_model = lmfit.Model(linear_function, independent_vars=["x"])
        xdata = np.linspace(0, 10, 40)
        data = 10 + 5 * xdata
        lmfit_model.fit(data, x=xdata, a=1, b=1, iter_cb=oc.lmfit_callback)
        plt.figure(100)
        plt.clf()
        oc.plot(logy=True)
        plt.close(100)

    def test_AverageDecreaseTermination(self):
        tc = AverageDecreaseTermination(4)
        results = [tc(0, None, value, None, None) for value in [4, 3, 2, 1, 0.1, 0, 0, 0, 0, 0.01, 0]]
        self.assertEqual(results, [False, False, False, False, False, False, False, False, False, True, True])

        tc = AverageDecreaseTermination(4, tolerance=1.0)
        results = [tc(0, None, value, None, None) for value in [4, 3, 2, 1, 0.1, 0, 0, 0, 0, 0.01, 1.0, 20.0]]
        self.assertEqual(results, [False, False, False, False, False, False, False, False, False, False, False, True])

    def test_AverageDecreaseTermination_properties(self):
        tc = AverageDecreaseTermination(3)
        tc(0, [1, 2], 10.0, None, None)
        tc(1, [2, 3], 9.0, None, None)

        self.assertEqual(tc.values, [10.0, 9.0])
        self.assertEqual(tc.parameters, [[1, 2], [2, 3]])

        tc.reset()
        self.assertEqual(tc.values, [])
        self.assertEqual(tc.parameters, [])

    def test_OptimizerCallback_optimization_time(self):
        oc = OptimizerCallback(show_progress=False)

        dt_empty = oc.optimization_time()
        self.assertEqual(dt_empty, 0)

        oc.data_callback(0, [1, 2], 10.0)
        import time

        time.sleep(0.05)
        oc.data_callback(1, [1.1, 2.1], 9.0)

        dt = oc.optimization_time()
        self.assertGreater(dt, 0.04)

    def test_OptimizerCallback_qiskit_callback(self):
        oc = OptimizerCallback(show_progress=False)
        oc.qiskit_callback(1, [0.5, 0.5], 1.5, 0.1, True)
        oc.qiskit_callback(2, [0.4, 0.6], 1.2, 0.1, True)

        self.assertEqual(oc.number_of_evaluations(), 2)
        self.assertEqual(len(oc.data), 2)
        self.assertEqual(oc.parameters, [[0.5, 0.5], [0.4, 0.6]])

    def test_OptimizerCallback_store_data_false(self):
        oc = OptimizerCallback(show_progress=False, store_data=False)
        oc.data_callback(0, [1, 2], 10.0)
        oc.data_callback(1, [1.1, 2.1], 9.0)

        self.assertEqual(oc.number_of_evaluations(), 2)
        self.assertEqual(len(oc.data), 0)

    def test_OptimizerLog_with_int_ax(self):
        log = OptimizerLog([1.0, 0.8, 0.5], [[0, 0], [0.1, 0.1], [0.2, 0.2]])
        log.plot(ax=99)
        plt.close(99)


if __name__ == "__main__":
    unittest.main()
