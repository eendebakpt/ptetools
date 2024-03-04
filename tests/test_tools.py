import io
import time
import unittest
from contextlib import redirect_stdout

from ptetools.tools import cprint, measure_time


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


def test_cprint():
    with redirect_stdout(io.StringIO()) as f:
        cprint("hi")
    assert f.getvalue() == "\x1b[36mhi\x1b[0m\n"


if __name__ == "__main__":
    unittest.main()
