"""

Copyright 2023 QuTech (TNO, TU Delft)

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

import datetime
import logging
from dataclasses import dataclass
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes


@dataclass
class OptimizerLog:
    values: list
    parameters: list

    def update(self, count, parameters, mean, _metadata):
        self.values.append(mean)
        self.parameters.append(parameters)
        print(f"Running circuit {count}: mean {mean:.2f}", end="\r\n", flush=True)

    def plot(self, ax=None):
        if ax is None:
            ax = plt.gca()
        if isinstance(ax, int):
            fig = plt.figure(ax)
            fig.clear()
            ax = plt.gca()
        # Plot energy and reference value
        ax.plot(self.values, label="Cost")

        # best_value
        # plt.axhline(y=-best_value, color="tab:red", ls="--", label="Target")

        ax.legend(loc="best")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")


class AverageDecreaseTermination:
    def __init__(self, N: int, tolerance: float = 0.0):
        """Callback to terminate optimization based the average decrease

        The average decrease over the last N data points is compared to the specified tolerance.
        The average decrease is determined by a linear fit (least squares) to the data.

        This class can be used as an argument to the Qiskit SPSA optimizer.

        Args:
            N: Number of data points to use
            tolerance: Abort if the average decrease is smaller than the specified tolerance

        """
        self.N = N
        self.tolerance = tolerance
        self.logger = logging.getLogger(self.__class__.__name__)
        self.reset()

    @property
    def parameters(self):
        return self._parameters

    @property
    def values(self):
        return self._values

    def reset(self):
        """Reset the data"""
        self._values = []
        self._parameters = []

    def __call__(self, nfev, parameters, value, update, accepted) -> bool:
        """
        Args:
            nfev: Number of evaluations
            parameters: Current parameters in the optimization
            value: Value of the objective function
            update: Update step
            accepted: Whether the update was accepted

        Returns:
            True if the optimization loop should be aborted
        """
        self._values.append(value)
        self._parameters.append(parameters)

        if len(self._values) > self.N:
            last_values = self._values[-self.N :]
            pp = np.polyfit(range(self.N), last_values, 1)
            slope = pp[0] / self.N

            self.logger.debug(f"AverageDecreaseTermination(N={self.N}): slope {slope}, tolerance {self.tolerance}")
            if slope > self.tolerance:
                self.logger.info(
                    f"AverageDecreaseTermination(N={self.N}): terminating with slope {slope},"
                    f" tolerance {self.tolerance}"
                )
                return True
        return False


class OptimizerCallback:
    _column_names = ["iteration", "timestamp", "residual"]

    def __init__(self, show_progress: bool = False, store_data: bool = True, residual_fitting: bool = True) -> None:
        """Class to collect data of optimization procedures

        The class contains methods that can be used as callbacks on several well-known optimization packages.

        Args:
            show_progress: If True, then print output for each iteration
            store_data: If True, store the callback data inside the object
            residual_fitting: If True, assume the optimizer is minimizing a residual

        """
        self.show_progress = show_progress
        self.store_data = store_data
        self.logger = logging.getLogger(self.__class__.__name__)
        self.clear()
        self._residual_fitting = residual_fitting

    @property
    def data(self) -> pd.DataFrame:
        """Return data gathered by callback"""

        df = pd.DataFrame(self._data, columns=self._column_names)

        return df

    @property
    def parameters(self) -> list[Any]:
        """Returns list of parameters that have been used in evaluations

        Returns:
            The list of parameters
        """
        return self._parameters

    def _append(self, d: tuple):
        """Append a row of data"""
        self._data.append(d)

    def clear(self):
        """Clear the data from this instance"""
        self._parameters = []
        self._data = []
        self._number_of_evaluations = 0

    def number_of_evaluations(self) -> int:
        """Return the number of callback evaluations

        Note: this can differ from the number of objective evaluations
        """
        return self._number_of_evaluations

    def optimization_time(self) -> float:
        """Return time difference between the first and the last invocation of the callback

        Returns:
            Time in seconds
        """
        if len(self.data) > 0:
            delta_t = self.data.iloc[-1]["timestamp"] - self.data.iloc[0]["timestamp"]
            dt = delta_t.total_seconds()
        else:
            dt = 0
        return dt

    def plot(self, ax: Axes | None = None, **kwargs) -> None:
        """Plot optimization results"""
        if ax is None:
            ax = plt.gca()

        self.data.plot("iteration", "residual", ax=ax, **kwargs)  # type: ignore
        dt = self.optimization_time()
        residual = self.data["residual"].iloc[-1]
        ax.set_title(f"Optimization total time {dt:.2f} [s], residual {residual:.2g}")
        if self._residual_fitting:
            ax.set_ylabel("Residual")
        else:
            ax.set_ylabel("Value")

    def data_callback(self, iteration: int, parameters: Any, residual: float) -> None:
        """Callback used to store data

        Args:
            iteration: Iteration on the optimization procedure
            parameters: Current values of the parameters to be optimized
            residual: Current residual (value of the objective function)

        """
        self._number_of_evaluations = self._number_of_evaluations + 1
        if self.store_data:
            self.logger.info(f"data_callback: {iteration} {parameters} {residual}")

            ts = datetime.datetime.now()  # .isoformat()
            d = (int(iteration), ts, float(residual))

            self.parameters.append(parameters)
            self._append(d)

    def qiskit_callback(self, number_evaluations, parameters, value, stepsize, accepted):
        """Callback method for Qiskit optimizers"""
        if self.show_progress:
            print(f"#{number_evaluations}, {parameters}, {value}, {stepsize}, {accepted}")
        self.data_callback(number_evaluations, parameters, value)

    def lmfit_callback(self, parameters, iteration, residual, *args, **kws):
        """Callback method for lmfit optimizers"""
        if self._residual_fitting:
            residual = np.linalg.norm(residual)

        if self.show_progress:
            print(f"#{iteration}, {parameters}, {residual}")
        self.data_callback(iteration, parameters, residual)

    def scipy_callback(self, parameters):
        """Callback method for scipy optimizers"""
        number_evaluations = self.number_of_evaluations()
        value = np.nan
        if self.show_progress:
            print(f"#{number_evaluations}, {parameters}")
        self.data_callback(number_evaluations, parameters, value)
