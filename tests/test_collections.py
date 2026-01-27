import unittest

from IPython.lib.pretty import pretty

from ptetools.collections import fnamedtuple


class TestCollections(unittest.TestCase):
    def test_fnamedtuple(self):
        point = fnamedtuple("Point", ["x", "y"])
        pt = point(2, 3)
        p = pretty(pt)
        assert "Point" in p

    def test_fnamedtuple_with_defaults(self):
        config = fnamedtuple("Config", ["name", "value"], defaults=["default"])
        cfg = config("test")
        assert cfg.name == "test"
        assert cfg.value == "default"

    def test_fnamedtuple_access(self):
        vector = fnamedtuple("Vector", ["x", "y", "z"])
        v = vector(1, 2, 3)
        assert v.x == 1
        assert v.y == 2
        assert v.z == 3
        assert v[0] == 1
        assert v[1] == 2
        assert v[2] == 3


if __name__ == "__main__":
    unittest.main()
