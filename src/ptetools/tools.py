import gc
import math
import operator
import os
import tempfile
import time
from collections.abc import Callable, Sequence
from itertools import chain, repeat
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import rich.pretty
from termcolor import colored

from ptetools._qtt import (  # noqa
    attribute_context,
    ginput,
    logging_context,
    measure_time,
    monitorSizes,
    robust_cost_function,
    setWindowRectangle,
    tilefigs,
)


def is_spyder_environment() -> bool:
    """Return True if the process is running in a Spyder environment"""
    return "SPY_TESTING" in os.environ


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
    X = np.asarray(X)
    assert len(X.shape) == 2, "input should be a 2-dimensional array"
    ss = ""
    if comment is not None:
        if isinstance(comment, list):
            for line in comment:
                ss += f"% {str(line)}\n"
        else:
            ss += f"% {str(comment)}\n"
    if header:
        match mode:
            case "tabular":
                if len(tabchar) == 1:
                    cc = tabchar * X.shape[1]
                else:
                    cc = tabchar + tabchar[-1] * (X.shape[1] - len(tabchar))
                ss += f"\\begin{{tabular}}{{{cc}}}" + chr(10)
            case "psmallmatrix":
                ss += "\\begin{psmallmatrix}" + chr(10)
            case "pmatrix":
                ss += "\\begin{pmatrix}" + chr(10)
            case _:
                raise ValueError(f"mode {mode} is invalid")
    for ii in range(X.shape[0]):
        r = X[ii, :]
        if isinstance(r[0], str):
            ss += " & ".join([f"{x}" for x in r])
        else:
            ss += " & ".join([floatfmt % x for x in r])
        if ii < (X.shape[0]) - 1 or not header:
            ss += "  \\\\" + chr(10)
        else:
            ss += "  " + chr(10)
        if ii in hlines:
            ss += r"\hline" + chr(10)
            if hlinespace is not None:
                ss += f"\\rule[+{hlinespace:.2f}ex]{{0pt}}{{0pt}}"
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


def sorted_dictionary(d: dict[Any, Any], *, key: Callable | None = None) -> dict[Any, Any]:
    """Sort keys of a dictionary"""
    return {k: d[k] for k in sorted(d, key=key)}


def cprint(s: str, color: str = "cyan", *args: Any, **kwargs: Any):
    """Colored print of string"""
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
        lbl: Sequence[str] = [f"{i}" for i in range(npoints)]
    else:
        lbl = labels
        if isinstance(lbl, (int, str)):
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

    rr: dict = {}
    for obj in gc.get_objects():
        tt = type(obj)
        rr[tt] = rr.get(tt, 0) + 1

    rr_many = {key: number for key, number in rr.items() if number > minimal_number_of_instances}
    rr_many = dict(sorted(rr_many.items(), key=operator.itemgetter(1), reverse=True))

    keys = list(rr_many.keys())
    results = {str(key): rr_many[key] for key in keys[:maximum_number_to_show]}
    if verbose:
        print("memory report:")
    for key, nn in results.items():
        if nn > 2000 and verbose:
            print(f"{key}: {nn}")

    return results


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
        if N <= 1:  # pragma: no cover
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


def interleaved_benchmark(
    func: Callable,
    func2: Callable,
    *args,
    target_duration: float = 1.0,
    **kwargs,
):
    t0 = time.perf_counter()
    for ii in range(1, 30):
        func(*args, **kwargs)
        dt = time.perf_counter() - t0
        if dt > 0.1:
            break
    if dt < 0.01:
        n_inner = 50
        t0 = time.perf_counter()
        for ii in range(1, 60):
            for ii in range(n_inner):
                func(*args, **kwargs)
            dt = time.perf_counter() - t0
            if dt > 0.1:
                break

        dt /= n_inner
    dt = dt / ii

    target_duration // (2 * dt)

    number_of_iterations = max(1, min(10_000_000, int(target_duration // (2 * dt))))
    # round to power of 10 or half of power of 10
    log_ten = math.log10(number_of_iterations)
    log_ten = (log_ten // 1) + (0 if (log_ten % 1) < math.log(5) else math.log(5))
    number_of_iterations = int(10**log_ten)

    dt1 = 0
    dt2 = 0

    if number_of_iterations > 1_000:
        blocksize = 200
    else:
        blocksize = 1

    if kwargs:
        if blocksize > 1:
            for ii in range(number_of_iterations // blocksize):
                t0 = time.perf_counter()
                for _ in repeat(None, blocksize):
                    func(*args, **kwargs)
                dt1 += time.perf_counter() - t0

                t0 = time.perf_counter()
                for _ in repeat(None, blocksize):
                    func2(*args, **kwargs)
                dt2 += time.perf_counter() - t0
        else:  # pragma: no cover
            for ii in range(number_of_iterations):
                t0 = time.perf_counter()
                func(*args, **kwargs)
                dt1 += time.perf_counter() - t0

                t0 = time.perf_counter()
                func2(*args, **kwargs)
                dt2 += time.perf_counter() - t0
    else:
        if len(args) == 0:
            if blocksize > 1:
                for ii in range(number_of_iterations // blocksize):
                    t0 = time.perf_counter()
                    for _ in repeat(None, blocksize):
                        func()
                    dt1 += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    for _ in repeat(None, blocksize):
                        func2()
                    dt2 += time.perf_counter() - t0
            else:  # pragma: no cover
                for ii in range(number_of_iterations):
                    t0 = time.perf_counter()
                    func()
                    dt1 += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    func2()
                    dt2 += time.perf_counter() - t0
        elif len(args) == 1:
            arg = args[0]

            if blocksize > 1:
                for ii in range(number_of_iterations // blocksize):
                    t0 = time.perf_counter()
                    for _ in repeat(None, blocksize):
                        func(arg)
                    dt1 += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    for _ in repeat(None, blocksize):
                        func2(arg)
                    dt2 += time.perf_counter() - t0
            else:  # pragma: no cover
                for ii in range(number_of_iterations):
                    t0 = time.perf_counter()
                    func(arg)
                    dt1 += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    func2(arg)
                    dt2 += time.perf_counter() - t0
        else:
            if blocksize > 1:
                for ii in range(number_of_iterations // blocksize):
                    t0 = time.perf_counter()
                    for _ in repeat(None, blocksize):
                        func(*args)
                    dt1 += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    for _ in repeat(None, blocksize):
                        func2(*args)
                    dt2 += time.perf_counter() - t0
            else:
                for ii in range(number_of_iterations):
                    t0 = time.perf_counter()
                    func(*args)
                    dt1 += time.perf_counter() - t0

                    t0 = time.perf_counter()
                    func2(*args)
                    dt2 += time.perf_counter() - t0

    gain = dt1 / dt2
    per_loop1 = (dt1 / number_of_iterations) * 1e6
    per_loop2 = (dt2 / number_of_iterations) * 1e6

    # format...
    gain_txt = f"gain {gain:.2f} ({number_of_iterations} loops)"
    if per_loop1 < 1:
        print(f"{per_loop1 * 1e3:.2f} ns ± ? μs per loop vs {per_loop2 * 1e3:.2f} ns, " + gain_txt)
    else:
        print(f"{per_loop1:.2f} μs ± ? μs per loop vs {per_loop2:.2f} μs, " + gain_txt)
    return dt1, dt2


if __name__ == "__main__":  # pragma: no cover

    def g(x):
        return x * x * x * x * x * x * x * x * x * x * x * x * x * x * x * x

    def h(x):
        y = x * x
        z = y * y
        w = z * z
        return w * w

    dt1, dt2 = interleaved_benchmark(g, h, 10)
    print(f"total {dt1 + dt2:.2f} [s]")


def _repr_pretty_rich_(self, p: Any, cycle: bool) -> None:
    del cycle
    z = rich.pretty.Pretty(self)
    s = (z._repr_mimebundle_([], []))["text/plain"]
    p.text(s)


# def add_rich_repr[T: type](cls: T) -> T:  # python 3.12+
def add_rich_repr(cls):
    """Add pretty representation method to a class using rich"""

    cls._repr_pretty_ = _repr_pretty_rich_  # ty: ignore
    return cls


# %%


class ReprPrettyTester:
    def __init__(self, obj=None, cycle: bool = False):
        self.txt = ""

        if obj is not None:
            obj._repr_pretty_(self, cycle=cycle)

    def text(self, v):
        self.txt += v

    def _repr_pretty_(self, p: Any, cycle: bool) -> None:
        del cycle
        p.text(f"{self.__class__.__name__}: {self.txt}")


if __name__ == "__main__":  # pragma: no cover
    from dataclasses import dataclass

    @add_rich_repr
    @dataclass
    class A:
        x: int = 10
        y: str = "hi"

    r = ReprPrettyTester(A())
    print(r.txt)
