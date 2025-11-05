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
import copy
import logging
import time
from collections import namedtuple
from collections.abc import Callable, Sequence
from math import cos, sin
from types import TracebackType
from typing import Any, Literal

import matplotlib
import matplotlib.figure
import matplotlib.pylab as pylab
import matplotlib.pyplot as plt
import numpy as np
import scipy

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


def monitorSizes(verbose: int = 0) -> list[tuple[int]]:  # pragma: no cover
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


@static_var("monitorindex", -1)  # pragma: no cover
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
        elif isinstance(f, int | np.int32 | np.int64):
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


def setWindowRectangle(  # pragma: no cover
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


def pg_scaling(scale: float | FloatArray, cc: FloatArray | None = None) -> FloatArray:
    """Create scale transformation with specified centre

    Args:
        scale: Scaling vector
        cc: Centre for the scale transformation. If None, then take the origin

    Returns:
        Scale transformation

    Example
    -------
    >>> pg_scaling( [1.,2])
    array([[ 1.,  0.,  0.],
           [ 0.,  2.,  0.],
           [ 0.,  0.,  1.]])

    """
    scale = np.hstack((scale, 1))
    H = np.diag(scale)
    if cc is not None:
        cc = np.asarray(cc).flatten()
        H = pg_transl2homogeneous(cc).dot(H).dot(pg_transl2homogeneous(-cc))

    return H


def hom(x: FloatArray) -> FloatArray:
    """Convert affine to homogeneous coordinates

    Args:
        x: k x N array in affine coordinates
    Returns:
        An (k+1xN) arrayin homogeneous coordinates
    """
    return np.vstack((x, np.ones_like(x, shape=(1, x.shape[1]))))


def dehom(x: np.ndarray) -> np.ndarray:
    """Convert homogeneous points to affine coordinates"""
    return x[0:-1, :] / x[-1, :]


def pg_transl2homogeneous(tr: FloatArray) -> FloatArray:
    """Convert translation to homogeneous transform matrix

    >>> pg_transl2homogeneous([1, 2])
    array([[ 1.,  0.,  1.],
            [ 0.,  1.,  2.],
            [ 0.,  0.,  1.]])

    """
    sh = np.asarray(tr)
    H = np.eye(sh.size + 1)
    H[0:-1, -1] = sh.flatten()
    return H


def pg_rotation2homogeneous(rotation_matrix: FloatArray) -> FloatArray:
    """Convert rotation matrix to homogenous transform matrix"""
    return pg_affine2hom(rotation_matrix)


def pg_rotx(phi: float) -> FloatArray:
    """Create rotation around the x-axis with specified angle"""
    c = cos(phi)
    s = sin(phi)
    R = np.zeros((3, 3))
    R.ravel()[:] = [1, 0, 0, 0, c, -s, 0, s, c]
    return R


def pg_rotz(phi: float) -> FloatArray:
    """Create rotation around the z-axis with specified angle"""
    c = cos(phi)
    s = sin(phi)
    R = np.zeros((3, 3))
    R.ravel()[:] = [c, -s, 0, s, c, 0, 0, 0, 1]
    return R


def mean_of_directions(vec):
    """Calculate the mean of a set of directions

    The initial direction is determined using the oriented direction. Then a non-linear optimization is done.

    Args:
        vec: List of directions

    Returns
        Angle of mean of directions

    >>> vv = np.array( [[1,0],[1,0.1], [-1,.1]])
    >>> a = mean_of_directions(vv)

    """
    vec = np.asarray(vec)
    vector_angles = np.arctan2(vec[:, 0], vec[:, 1])

    mod = np.mod
    norm = np.linalg.norm

    def cost_function(a):
        x = mod(a - vector_angles + np.pi / 2, np.pi) - np.pi / 2
        cost = norm(x)
        return cost

    m = vec.mean(axis=0)
    angle_initial_guess = np.arctan2(m[0], m[1])

    r = scipy.optimize.minimize(cost_function, angle_initial_guess, callback=None, options=({"disp": False}))
    angle = r.x[0]
    return angle


def pg_affine2homogeneous(affine_transform: FloatArray) -> FloatArray:
    """Create homogeneous transformation from affine transformation

    Args:
        U: Affine transformation

    Returns:
        Homogeneous transformation

    Example
    -------
    >>> pg_affine2homogeneous(np.array([[2.]]))
    array([[ 2.,  0.],
           [ 0.,  1.]])

    """
    H = np.eye(affine_transform.shape[0] + 1, dtype=affine_transform.dtype)
    H[:-1, :-1] = affine_transform
    return H


pg_affine2hom = pg_affine2homogeneous


def projective_transformation(H: FloatArray, x: FloatArray) -> FloatArray:
    """Apply a projective transformation to a k x N array

    >>> y = projective_transformation(np.eye(3), np.random.rand( 2, 10))
    """
    try:
        import cv2
    except ImportError:
        return dehom(H @ hom(x))

    k = x.shape[0]
    kout = H.shape[0] - 1
    xx = x.transpose().reshape((-1, 1, k))

    if xx.dtype is np.integer or xx.dtype == "int64":
        xx = xx.astype(np.float32)
    if xx.size > 0:
        ww = cv2.perspectiveTransform(xx, H)
        ww = ww.reshape((-1, kout)).transpose()
        return ww
    else:
        return copy.copy(x)


def decompose_projective_transformation(
    H: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray, tuple[Any, Any, FloatArray, FloatArray]]:
    """Decompose projective transformation

    H is decomposed as H = Hs * Ha * Hp with

     Hs = [sR t]
          [0  1]

     Ha = [K 0]
          [0 1]

     Hp = [I 0]
          [v' eta]

    If H is 3-dimensional, then R = [cos(phi) -sin(phi); sin(phi) cos(phi)];

    For more information see "Multiple View Geometry", paragraph 1.4.6.

    >>> Ha, Hs, Hp, rest = decomposeProjectiveTransformation( np.eye(3) )
    """
    H = np.asarray(H)
    k = H.shape[0]
    km = k - 1

    eta = H[k - 1, k - 1]
    Hprojective = np.vstack((np.eye(km, k), H[k - 1, :]))
    A = H[0:km, 0:km]
    t = H[0:km, -1]
    v = H[k - 1, 0:km].T

    eps = 1e-10
    if np.abs(np.linalg.det(A)) < 4 * eps:
        print("decompose_projective_transformation: part A of matrix is (near) singular")

    sRK = A - np.array(t).dot(np.array(v.T))
    # upper left block of H*inv(Hprojective)
    R, K = np.linalg.qr(sRK)
    K = np.asarray(K)
    R = np.asarray(R)

    s = (np.abs(np.linalg.det(K))) ** (1.0 / km)
    K = K / s

    if k == 2 and K[0, 0] < 0:  # in 3-dimensional case normalize sign
        K = np.diag([-1, 1]) * K
        R = R.dot(np.diag([-1, 1]))
    else:
        # primitive...
        sc = np.sign(np.diag(K))
        K = np.diag(sc).dot(K)
        R = R.dot(np.diag(sc))
    br = np.hstack((np.zeros((1, km)), np.ones((1, 1))))
    Hs = np.vstack((np.hstack((s * R, t.reshape((-1, 1)))), br))
    Ha = np.vstack((np.hstack((K, np.zeros((km, 1)))), br))

    phi = np.arctan2(R[1, 0], R[0, 0])

    elements = namedtuple("elements", ["s", "phi", "t", "v", "eta"])
    rest = elements(s, phi, t, v, eta)
    return Ha, Hs, Hprojective, rest
