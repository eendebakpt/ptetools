import unittest

from ptetools.computation import parallel_execute


class TestTools(unittest.TestCase):
    def test_parallel_execute(self):
        def method(x):
            return x * x

        data = [{"x": ii} for ii in range(4)]
        results = parallel_execute(method, data)
        assert results == [0, 1, 4, 9]


if __name__ == "__main__":
    unittest.main()
