import os
import tempfile
import time
from collections.abc import Callable, Sequence
from itertools import chain
from types import TracebackType
from typing import Any, Literal

import matplotlib
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np


def is_spyder_environment() -> bool:
    """Return True if the process is running in a Spyder environment"""
    return "SPY_TESTING" in os.environ


if is_spyder_environment():
    pass
else:
    pass


def fmt_dict(d: dict[Any, Any], fmt: str = "{:.2f}", *, key_fmt: str = "{}", add_braces: bool = True) -> str:
    """Format dictionary keys and values"""
    body = ", ".join([f"{key_fmt.format(k)}: {fmt.format(v)}" for k, v in d.items()])
    if add_braces:
        return "{" + body + "}"
    else:
        return body


def array2latex(
    X,
    header: bool = True,
    hlines=(),
    floatfmt: str = "%g",
    comment: str | None = None,
    hlinespace: None | float = None,
    mode: Literal["tabular", "psmallmatrix", "pmatrix"] = "tabular",
    tabchar: str = "c",
) -> str:
    """Convert numpy array to Latex tabular or matrix"""
    ss = ""
    if comment is not None:
        if isinstance(comment, list):
            for line in comment:
                ss += "%% %s\n" % str(line)
        else:
            ss += "%% %s\n" % str(comment)
    if header:
        match mode:
            case "tabular":
                if len(tabchar) == 1:
                    cc = tabchar * X.shape[1]
                else:
                    cc = tabchar + tabchar[-1] * (X.shape[1] - len(tabchar))
                ss += "\\begin{tabular}{%s}" % cc + chr(10)
            case "psmallmatrix":
                ss += "\\begin{psmallmatrix}" + chr(10)
            case "pmatrix":
                ss += "\\begin{pmatrix}" + chr(10)
            case _:
                raise ValueError(f"mode {mode} is invalid")
    for ii in range(X.shape[0]):
        r = X[ii, :]
        if isinstance(r[0], str):
            ss += " & ".join(["%s" % x for x in r])
        else:
            ss += " & ".join([floatfmt % x for x in r])
        if ii < (X.shape[0]) - 1 or not header:
            ss += "  \\\\" + chr(10)
        else:
            ss += "  " + chr(10)
        if ii in hlines:
            ss += r"\hline" + chr(10)
            if hlinespace is not None:
                ss += "\\rule[+%.2fex]{0pt}{0pt}" % hlinespace
    if header:
        match mode:
            case "tabular":
                ss += "\\end{tabular}"
            case "psmallmatrix":
                ss += "\\end{psmallmatrix}" + chr(10)
            case "pmatrix":
                ss += "\\end{pmatrix}" + chr(10)
            case _:
                raise ValueError(f"mode {mode} is invalid")
    return ss


def flatten(lst: Sequence[Any]) -> list[Any]:
    """Flatten a sequence.

    Args:
        lst : Sequence to be flattened.

    Returns:
        list: flattened list.

    Example:
        >>> flatten([ [1,2], [3,4], [10] ])
        [1, 2, 3, 4, 10]
    """
    return list(chain(*lst))


def make_blocks(size: int, block_size: int) -> list[tuple[int, int]]:
    """Create blocks of specified size"""
    number_of_blocks = (size + block_size - 1) // block_size
    blocks = [(ii * block_size, min(size, (ii + 1) * block_size)) for ii in range(number_of_blocks)]
    return blocks


def sorted_dictionary(d: dict[Any, Any], *, key: Callable = None) -> dict[Any, Any]:
    """Sort keys of a dictionary"""
    return {k: d[k] for k in sorted(d, key=key)}


def cprint(s: str, color: str = "cyan", *args, **kwargs):
    """Colored print of string"""
    from termcolor import colored

    print(colored(s, color=color), *args, **kwargs)


def plotLabels(points, labels: None | Sequence[str] = None, **kwargs: Any):
    """Plot labels next to points

    Args:
        xx (2xN array): Positions to plot the labels
        labels: Labels to plot
        *kwargs: arguments past to plotting function
    Example:
    >>> points = np.random.rand(2, 10)
    >>> fig=plt.figure(10); plt.clf()
    >>> _ = plotPoints(points, '.'); _ = plotLabels(points)
    """

    points = np.asarray(points)
    if len(points.shape) == 1 and points.shape[0] == 2:
        points = points.reshape((2, 1))
    npoints = points.shape[1]

    if labels is None:
        lbl: Sequence[str] = ["%d" % i for i in range(npoints)]
    else:
        lbl = labels
        if isinstance(lbl, int):
            lbl = [str(lbl)]
        elif isinstance(lbl, str):
            lbl = [str(lbl)]
    ax = plt.gca()
    th: list[Any] = [None] * npoints
    for ii in range(npoints):
        lbltxt = str(lbl[ii])
        th[ii] = ax.annotate(lbltxt, points[:, ii], **kwargs)
    return th


def memory_report(
    maximum_number_to_show: int = 24, minimal_number_of_instances: int = 100, verbose: bool = True
) -> dict[str, int]:
    """Show information about objects with most occurences in memory

    For a more detailed analysis: check the heapy package (https://github.com/zhuyifei1999/guppy3/)
    """
    import gc
    import operator

    rr: dict = {}
    for obj in gc.get_objects():
        tt = type(obj)
        rr[tt] = rr.get(tt, 0) + 1

    rr_many = {key: number for key, number in rr.items() if number > minimal_number_of_instances}
    rr_many = {key: value for key, value in sorted(rr_many.items(), key=operator.itemgetter(1), reverse=True)}

    keys = list(rr_many.keys())

    if verbose:
        print("memory report:")
    for key in keys[:maximum_number_to_show]:
        nn = rr_many[key]
        if nn > 2000:
            if verbose:
                print(f"{key}: {nn}")

    results = {str(key): value for key, value in rr_many.items()}
    return results


""" Code below is derived from QTT

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
    import qtpy.QtWidgets  # lazy import

    _ = qtpy.QtWidgets.QApplication.instance()  # type: ignore
    _qd = qtpy.QtWidgets.QDesktopWidget()  # type: ignore

    nmon = _qd.screenCount()
    monitor_rectangles = [_qd.screenGeometry(ii) for ii in range(nmon)]
    monitor_sizes: list[tuple[int]] = [(w.x(), w.y(), w.width(), w.height()) for w in monitor_rectangles]  # type: ignore

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
        x: int = int(ww[0]) + int(ix * w)  # type: ignore
        y: int = int(ww[1]) + int(iy * h)  # type: ignore
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
        elif be in ("Qt4Agg", "QT4", "QT5Agg", "Qt5Agg", "QtAgg", "qtagg"):
            # assume Qt canvas
            try:
                # fig.canvas.manager.window.move(x, y+y_offset)  # type: ignore
                # fig.canvas.manager.window.resize(int(w), int(h))  # type: ignore
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


# %%
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


if __name__ == "__main__":  # pragma: no cover
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
    if y is None:
        x, y, w, h = x  # type: ignore
    if mngr is None:
        mngr = plt.get_current_fig_manager()
    be = matplotlib.get_backend()
    if be == "WXAgg":
        mngr.canvas.manager.window.SetPosition((x, y))
        mngr.canvas.manager.window.SetSize((w, h))
    elif be == "TkAgg":
        _ = mngr.canvas.manager.window.wm_geometry("%dx%d+%d+%d" % (w, h, x, y))  # type: ignore
    elif be == "module://IPython.kernel.zmq.pylab.backend_inline":
        pass
    else:
        # assume Qt canvas
        mngr.canvas.manager.window.move(x, y)
        mngr.canvas.manager.window.resize(w, h)
        mngr.canvas.manager.window.setGeometry(x, y, w, h)
