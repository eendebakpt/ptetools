from collections import namedtuple
from functools import wraps
from typing import Any

from ptetools.tools import add_rich_repr


@wraps(namedtuple)
def fnamedtuple(*args: Any, **kwargs: Any):
    """Named tuple with rich representation"""
    n = namedtuple(*args, **kwargs)
    return add_rich_repr(n)


if __name__ == "__main__":  # pragma: no cover
    from IPython.lib.pretty import pretty

    Point = fnamedtuple("Point", ["x", "y"])
    pt = Point(2, 3)
    print(pretty(pt))
