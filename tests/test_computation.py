import unittest

from ptetools.computation import make_blocks, parallel_execute


class TestTools(unittest.TestCase):
    def test_parallel_execute(self):
        def method(x):
            return x * x

        data = [{"x": ii} for ii in range(4)]
        results = parallel_execute(method, data)
        assert results == [0, 1, 4, 9]

    def test_parallel_execute_no_progress_bar(self):
        def method(x):
            return x + 1

        data = [{"x": ii} for ii in range(3)]
        results = parallel_execute(method, data, progress_bar="")
        assert results == [1, 2, 3]

    def test_parallel_execute_custom_block_size(self):
        def method(x):
            return x * 2

        data = [{"x": ii} for ii in range(10)]
        results = parallel_execute(method, data, block_size=2)
        assert results == [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]

    def test_parallel_execute_single_job(self):
        def method(x):
            return x

        data = [{"x": ii} for ii in range(5)]
        results = parallel_execute(method, data, n_jobs=1)
        assert results == [0, 1, 2, 3, 4]

    def test_parallel_execute_empty_data(self):
        def method(x):
            return x

        data = []
        results = parallel_execute(method, data)
        assert results == []

    def test_make_blocks(self):
        assert make_blocks(10, 3) == [(0, 3), (3, 6), (6, 9), (9, 10)]
        assert make_blocks(6, 2) == [(0, 2), (2, 4), (4, 6)]
        assert make_blocks(5, 10) == [(0, 5)]
        assert make_blocks(0, 5) == []


if __name__ == "__main__":
    unittest.main()
