from collections import namedtuple
from functools import wraps

from ptetools.tools import add_rich_repr


@wraps(namedtuple)
def fnamedtuple(*args, **kwargs):
    n = namedtuple(*args, **kwargs)
    return add_rich_repr(n)


if __name__ == "__main__":
    from IPython.lib.pretty import pretty

    Point = fnamedtuple("Point", ["x", "y"])
    pt = Point(2, 3)
    print(pretty(pt))
