"""Code below is derived from QTT

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

import contextlib
import logging
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from types import TracebackType
from typing import Any, Literal

import matplotlib
import matplotlib.figure
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np

FloatArray = np.typing.NDArray[np.float64]


def robust_cost_function(x: FloatArray, thr: None | float | str, method: str = "L1") -> FloatArray | list[str]:
    """Robust cost function

    For details see "Multiple View Geometry in Computer Vision, Second Edition", Hartley and Zisserman, 2004

    Args:
       x: data to be transformed
       thr: threshold. If None then the input x is returned unmodified. If 'auto' then use automatic detection
            (at 95th percentile)
       method : method to be used. Use 'show' to show the options

    Returns:
        Cost for each element in the input array

    Example
    -------
    >>> robust_cost_function([2, 3, 4], thr=2.5)
    array([ 2. ,  2.5,  2.5])
    >>> robust_cost_function(2, thr=1)
    1
    >>> methods=robust_cost_function(np.arange(-5,5,.2), thr=2, method='show')
    """
    if thr is None:
        return x

    if thr == "auto":
        ax = np.abs(x)
        p50, thr, p99 = np.percentile(ax, [50, 95.0, 99])
        assert isinstance(thr, float)

        if thr == p50:
            thr = p99
        if thr <= 0:
            thr = np.mean(ax)

        if method == "L2" or method == "square":
            thr = thr * thr

    assert not isinstance(thr, str)

    match method:
        case "L1":
            y = np.minimum(np.abs(x), thr)
        case "L2" | "square":
            y = np.minimum(x * x, thr)
        case "BZ":
            alpha = thr * thr
            epsilon = np.exp(-alpha)
            y = -np.log(np.exp(-x * x) + epsilon)
        case "BZ0":
            alpha = thr * thr
            epsilon = np.exp(-alpha)
            y = -np.log(np.exp(-x * x) + epsilon) + np.log(1 + epsilon)
        case "cauchy":
            b2 = thr * thr
            d2 = x * x
            y = np.log(1 + d2 / b2)
        case "cg":
            delta = x
            delta2 = delta * delta
            w = 1.0 / thr  # ratio of std.dev
            w2 = w * w
            A = 0.1  # fraction of outliers
            y = -np.log(A * np.exp(-delta2) + (1 - A) * np.exp(-delta2 / w2) / w)
            y = y + np.log(A + (1 - A) * 1 / w)
        case "huber":
            d2 = x * x
            d = 2 * thr * np.abs(x) - thr * thr
            y = d2
            idx = np.abs(y) >= thr * thr
            y[idx] = d[idx]
        case "show":
            plt.figure(10)
            plt.clf()
            method_names = ["L1", "L2", "BZ", "cauchy", "huber", "cg"]
            for m in method_names:
                plt.plot(x, robust_cost_function(x, thr, m), label=m)
            plt.legend()
            return method_names
        case _:
            raise ValueError(f"no such method {method}")
    return y


def monitorSizes(verbose: int = 0) -> list[tuple[int]]:
    """Return monitor sizes

    Args:
        verbose: Verbosity level
    Returns:
        List with for each screen a list x, y, width, height
    """
    import qtpy.QtWidgets  # lazy import

    _ = qtpy.QtWidgets.QApplication.instance()  # type: ignore
    _qd = qtpy.QtWidgets.QDesktopWidget()  # type: ignore

    nmon = _qd.screenCount()
    monitor_rectangles = [_qd.screenGeometry(ii) for ii in range(nmon)]
    monitor_sizes: list[tuple[int]] = [(w.x(), w.y(), w.width(), w.height()) for w in monitor_rectangles]  # type: ignore

    if verbose:
        for ii, w in enumerate(monitor_sizes):
            print(f"monitor {ii}: {w}")
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
    ww: tuple[int] | list[int] | None = None,
    raisewindows: bool = False,
    tofront: bool = False,
    verbose: int = 0,
    monitorindex: int | None = None,
    y_offset: int = 20,
    window: tuple[int] | None = None,
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
        y_offset: Offset for window tile bars
    """

    if geometry is None:
        geometry = (2, 2)
    mngr = plt.get_current_fig_manager()
    be = matplotlib.get_backend()
    if monitorindex is None:
        monitorindex = tilefigs.monitorindex

    if ww is None:
        ww = monitorSizes()[monitorindex]

    if window is not None:
        ww = window

    w = ww[2] / geometry[0]  # type: ignore
    h = ww[3] / geometry[1]  # type: ignore

    if isinstance(lst, int):
        lst = [lst]
    elif isinstance(lst, np.ndarray):  # ty: ignore
        lst = lst.flatten().astype(int)  # ty: ignore

    if verbose:
        print(f"tilefigs: ww {ww}, w {w} h {h}")
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
        x: int = int(ww[0]) + int(ix * w)  # type: ignore
        y: int = int(ww[1]) + int(iy * h)  # type: ignore
        if be == "WXAgg" or be == "WX":
            fig.canvas.manager.window.SetPosition((x, y))  # type: ignore
            fig.canvas.manager.window.SetSize((w, h))  # type: ignore
        elif be == "agg":
            fig.canvas.manager.window.SetPosition((x, y))  # type: ignore
            fig.canvas.manager.window.resize(w, h)  # type: ignore
        elif be in ("Qt4Agg", "QT4", "QT5Agg", "Qt5Agg", "QtAgg", "qtagg"):
            # assume Qt canvas
            try:
                fig.canvas.manager.window.setGeometry(x, y + y_offset, int(w), int(h))  # type: ignore
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

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        del cycle
        s = f"<{self.__class__.__name__} at 0x{id(self):x}: dt {self.delta_time:.3f}>\n"
        p.text(s)


class NoValue:
    pass


class attribute_context:
    no_value = NoValue()

    def __init__(self, obj, attrs: None | dict[str, Any] = None, **kwargs):
        """Context manager to update attributes of an object

        Example:
            >>> import sys
            >>> with attribute_context(sys, copyright = 'Python license'):
            >>>     pass
        """
        self.obj = obj
        if attrs is None:
            attrs = {}
        self.kwargs = attrs | kwargs
        self.original = None

    def __enter__(self) -> "attribute_context":
        self.original = {key: getattr(self.obj, key) for key in self.kwargs}
        for key, value in self.kwargs.items():
            if value is not self.no_value:
                setattr(self.obj, key, value)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_traceback: TracebackType | None,
    ) -> Literal[False]:
        for key, value in self.original.items():
            setattr(self.obj, key, value)
        self.original = None
        return False


# %%
def profile_expression(expression: str, N: int | None = 1, gui: None | str = "snakeviz") -> tuple[str, Any]:
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
    if gui is not None:
        r = subprocess.Popen([gui, statsfile])
    else:
        r = None
    return statsfile, r


def ginput(number_of_points=1, marker: str | None = ".", linestyle="", **kwargs):  # pragma: no cover
    """Select points from matplotlib figure

    Press middle mouse button to stop selection

    Arguments:
        number_of_points: number of points to select
        marker: Marker style for plotting. If None, do not plot
        kwargs : Arguments passed to plot function
    Returns:
        Numpy array with selected points
    """
    kwargs = {"linestyle": ""} | kwargs
    xx = np.ones((number_of_points, 2)) * np.nan
    for ii in range(number_of_points):
        x = pylab.ginput(1)
        if len(x) == 0:
            break
        x = np.asarray(x)
        xx[ii, :] = x.flat
        if marker is not None:
            plt.plot(xx[: ii + 1, 0].T, xx[: ii + 1, 1].T, marker=marker, **kwargs)
            plt.draw()
    plt.pause(1e-3)
    return xx


if __name__ == "__main__" and 0:  # pragma: no cover
    plt.figure(10)
    plt.clf()
    plt.plot([0, 1, 2, 3], [0, 3, 1, 3], ".-")
    plt.draw()
    x = ginput(7)


def setWindowRectangle(
    x: int | Sequence[int],
    y: int | None = None,
    w: int | None = None,
    h: int | None = None,
    fig: int | None = None,
    mngr=None,
):
    """Position the current Matplotlib figure at the specified position

    Args:
        x: position in format (x,y,w,h)
        y, w, h: y position, width, height
        fig: specification of figure window. Use None for the current active window

    Usage: setWindowRectangle([x, y, w, h]) or setWindowRectangle(x, y, w, h)
    """
    if isinstance(fig, int):
        plt.figure(fig)

    if y is None:
        x, y, w, h = x  # type: ignore
    if mngr is None:
        mngr = plt.get_current_fig_manager()
    be = matplotlib.get_backend()
    if be == "WXAgg":
        mngr.canvas.manager.window.SetPosition((x, y))  # ty: ignore
        mngr.canvas.manager.window.SetSize((w, h))  # ty: ignore
    elif be == "TkAgg":
        _ = mngr.canvas.manager.window.wm_geometry(f"{w}x{h}x+{x}+{y}")  # type: ignore
    elif be == "module://IPython.kernel.zmq.pylab.backend_inline":
        pass
    else:
        # assume Qt canvas
        mngr.canvas.manager.window.move(x, y)  # ty: ignore
        mngr.canvas.manager.window.resize(w, h)  # ty: ignore
        mngr.canvas.manager.window.setGeometry(x, y, w, h)  # ty: ignore


@contextlib.contextmanager
def logging_context(level: int = logging.INFO, logger: None | logging.Logger = None):
    """A context manager that changes the logging level

    Args:
        level: Logging level to set in the context
        logger: Logger to update, if None then update the default logger

    """
    if logger is None:
        logger = logging.getLogger()
    previous_level = logger.getEffectiveLevel()
    logger.setLevel(level)

    try:
        yield
    finally:
        logger.setLevel(previous_level)
