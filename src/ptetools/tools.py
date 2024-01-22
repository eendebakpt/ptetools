import os
import tempfile
import time
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Any, Literal

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import qtpy

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


def monitorSizes(verbose: int = 0) -> list[tuple[int]]:
    """Return monitor sizes

    Args:
        verbose: Verbosity level
    Returns:
        List with for each screen a list x, y, width, height
    """
    _ = qtpy.QtWidgets.QApplication.instance()

    _qd = qtpy.QtWidgets.QDesktopWidget()

    nmon = _qd.screenCount()
    monitor_rectangles = [_qd.screenGeometry(ii) for ii in range(nmon)]
    monitor_sizes: list[tuple[int]] = [(w.x(), w.y(), w.width(), w.height()) for w in monitor_rectangles]

    if verbose:
        for ii, w in enumerate(monitor_sizes):
            print("monitor %d: %s" % (ii, str(w)))
    return monitor_sizes


def static_var(variable_name: str, value: Any) -> Callable:
    """Helper method to create a static variable on an object

    Args:
        variable_name: Variable to create
        value: Initial value to set
    """

    def static_variable_decorator(func):
        setattr(func, variable_name, value)
        return func

    return static_variable_decorator


@static_var("monitorindex", -1)
def tilefigs(
    lst: list[int | plt.Figure],
    geometry: Sequence[int] | None = None,
    ww: list[int] | None = None,
    raisewindows: bool = False,
    tofront: bool = False,
    verbose: int = 0,
    monitorindex: int | None = None,
) -> None:
    """Tile figure windows on a specified area

    Arguments
    ---------
        lst: list of figure handles or integers
        geometry: 2x1 array, layout of windows
        ww: monitor sizes
        raisewindows: When True, request that the window be raised to appear above other windows
        tofront: When True, activate the figure
        verbose: Verbosity level
        monitorindex: index of monitor to use for output

    """
    if geometry is None:
        geometry = [2, 2]
    mngr = plt.get_current_fig_manager()
    be = matplotlib.get_backend()
    if monitorindex is None:
        monitorindex = tilefigs.monitorindex

    if ww is None:
        ww = monitorSizes()[monitorindex]

    w = ww[2] / geometry[0]
    h = ww[3] / geometry[1]

    if isinstance(lst, int):
        lst = [lst]
    elif isinstance(lst, np.ndarray):
        lst = lst.flatten().astype(int)

    if verbose:
        print("tilefigs: ww %s, w %d h %d" % (str(ww), w, h))
    for ii, f in enumerate(lst):
        if isinstance(f, matplotlib.figure.Figure):
            fignum = f.number  # type: ignore
        elif isinstance(f, (int, np.int32, np.int64)):
            fignum = f
        else:
            try:
                fignum = f.fig.number
            except BaseException:
                fignum = -1
        if not plt.fignum_exists(fignum) and verbose >= 2:
            print(f"tilefigs: f {f} fignum: {str(fignum)}")
        fig = plt.figure(fignum)
        iim = ii % np.prod(geometry)
        ix = iim % geometry[0]
        iy = int(np.floor(float(iim) / geometry[0]))
        x: int = int(ww[0]) + int(ix * w)
        y: int = int(ww[1]) + int(iy * h)
        if verbose:
            print("ii %d: %d %d: f %d: %d %d %d %d" % (ii, ix, iy, fignum, x, y, w, h))
        if be == "WXAgg":
            fig.canvas.manager.window.SetPosition((x, y))  # type: ignore
            fig.canvas.manager.window.SetSize((w, h))  # type: ignore
        elif be == "WX":
            fig.canvas.manager.window.SetPosition((x, y))  # type: ignore
            fig.canvas.manager.window.SetSize((w, h))  # type: ignore
        elif be == "agg":
            fig.canvas.manager.window.SetPosition((x, y))  # type: ignore
            fig.canvas.manager.window.resize(w, h)  # type: ignore
        elif be == "Qt4Agg" or be == "QT4" or be == "QT5Agg" or be == "Qt5Agg":
            # assume Qt canvas
            try:
                fig.canvas.manager.window.move(x, y)  # type: ignore
                fig.canvas.manager.window.resize(int(w), int(h))  # type: ignore
                fig.canvas.manager.window.setGeometry(x, y, int(w), int(h))  # type: ignore
            except Exception as e:
                print(
                    "problem with window manager: ",
                )
                print(be)
                print(e)
        else:
            raise NotImplementedError(f"unknown backend {be}")
        if raisewindows:
            mngr.window.raise_()  # type: ignore
        if tofront:
            plt.figure(f)


class measure_time:
    """Create context manager that measures execution time and prints to stdout

    Example:
        >>> import time
        >>> with measure_time():
        ...     time.sleep(.1)
    """

    def __init__(self, message: str | None = "dt: "):
        self.message = message
        self.dt = float("nan")

    def __enter__(self) -> "measure_time":
        self.start_time = time.perf_counter()
        return self

    @property
    def current_delta_time(self) -> float:
        """Return time since start of the context

        Returns:
            Time in seconds
        """
        return time.perf_counter() - self.start_time

    @property
    def delta_time(self) -> float:
        """Return time spend in the context

        If still in the context, return nan.

        Returns:
            Time in seconds
        """
        return self.dt

    def __exit__(  # pylint: disable-all
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> Literal[False]:
        self.dt = time.perf_counter() - self.start_time

        if self.message is not None:
            print(f"{self.message} {self.dt:.3f} [s]")

        return False


def profile_expression(expression: str, N: int | None = 1, gui: str = "snakeviz") -> tuple[str, Any]:
    """Profile an expression with cProfile and display the results using snakeviz

    Args:
        expression: Code to be profiled
        N: Number of iterations. If None, then automatically determine a suitable number of iterations
        gui: Can be `tuna` or `snakeviz`
    Returns:
        Tuple with the filename of the profiling results and a handle to the subprocess starting the GUI
    """
    import cProfile  # lazy import
    import subprocess

    tmpdir = tempfile.mkdtemp()
    statsfile = os.path.join(tmpdir, "profile_expression_stats")

    assert isinstance(expression, str), "expression should be a string"

    if N is None:
        t0 = time.perf_counter()
        cProfile.run(expression, filename=statsfile)
        dt = time.perf_counter() - t0
        N = int(1.0 / max(dt - 0.6e-3, 1e-6))
        if N <= 1:
            print(f"profiling: 1 iteration, {dt:.2f} [s]")
            r = subprocess.Popen([gui, statsfile])
            return statsfile, r
    else:
        N = int(N)
    print(f"profile_expression: running {N} loops")
    if N > 1:
        loop_expression = f"for ijk_kji_no_name in range({N}):\n"
        loop_expression += "\n".join(["  " + term for term in expression.split("\n")])
        loop_expression += "\n# loop done"
        expression = loop_expression
    t0 = time.perf_counter()
    cProfile.run(expression, statsfile)
    dt = time.perf_counter() - t0

    print(f"profiling: {N} iterations, {dt:.2f} [s]")
    r = subprocess.Popen([gui, statsfile])

    return statsfile, r
