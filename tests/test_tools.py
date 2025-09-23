import io
import time
import unittest
from contextlib import redirect_stdout

import matplotlib.pyplot as plt
import numpy as np
import pytest
from IPython.lib.pretty import pretty

from ptetools.tools import (
    add_rich_repr,
    attribute_context,
    cprint,
    make_blocks,
    measure_time,
    memory_report,
    plotLabels,
    profile_expression,
    robust_cost_function,
    sorted_dictionary,
)


class TestTools(unittest.TestCase):
    def test_measure_time(self):
        with redirect_stdout(io.StringIO()) as f:
            with measure_time("hi") as m:
                time.sleep(0.101)
        self.assertGreater(m.delta_time, 0.1)
        self.assertIn("hi", f.getvalue())

    def test_measure_time_current_delta_time(self):
        with measure_time(None) as m:
            self.assertIsInstance(m.current_delta_time, float)
            self.assertTrue(m.current_delta_time >= 0, "current time must always be positive")

    def sorted_dictionary(self):
        d = sorted_dictionary({"b": 0, "a": 2})
        assert str(d) == "{'a': 2, 'b': 0}"

        with pytest.raises(TypeError):
            sorted_dictionary(10)

    def test_make_blocks(self):
        assert make_blocks(5, 2) == [(0, 2), (2, 4), (4, 5)]
        assert make_blocks(3, 4) == [(0, 3)]
        assert make_blocks(0, 4) == []
        with pytest.raises(ZeroDivisionError):
            make_blocks(0, 0)

    def test_plotLabels(self):
        plt.figure(1)
        plotLabels([[0, 1, 4, 5], [2, 3, 2, 3]])
        plt.close(1)

    def test_memory_report(self):
        x = memory_report(6, verbose=False)
        assert "<class 'tuple'>" in x

    def test_profile_expression(self):
        _ = profile_expression("import time", gui=None)


def test_attribute_context():
    import sys

    value = sys.api_version
    with attribute_context(sys, api_version=10):
        assert sys.api_version == 10
    assert sys.api_version == value


def test_cprint():
    with redirect_stdout(io.StringIO()) as f:
        cprint("hi")
    value = f.getvalue()
    assert value == "\x1b[36mhi\x1b[0m\n" or value == "hi\n"


def test_add_rich_repr():
    @add_rich_repr
    class AAA:
        pass

    a = AAA()
    assert "AAA" in pretty(a)


def test_robust_cost_function():
    x = np.linspace(0, 2, 11)
    methods = robust_cost_function(x, 5, "show")
    for method in methods:
        _ = robust_cost_function(x, "auto", method=method)

    x = np.array([0, 0.1, 0.5, 1])
    np.testing.assert_almost_equal(robust_cost_function(x, 0.5, "L1"), [0.0, 0.1, 0.5, 0.5])
    np.testing.assert_almost_equal(robust_cost_function(x, 0.5, "L2"), [0.0, 0.01, 0.25, 0.5])
    np.testing.assert_almost_equal(robust_cost_function(x, 0.5, "cauchy"), [0.0, 0.03922071, 0.69314718, 1.60943791])


if __name__ == "__main__":
    unittest.main()
