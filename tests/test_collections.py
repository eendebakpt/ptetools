import unittest

from IPython.lib.pretty import pretty

from ptetools.collections import fnamedtuple


class TestCollections(unittest.TestCase):
    def test_fnamedtuple(self):
        Point = fnamedtuple("Point", ["x", "y"])
        pt = Point(2, 3)
        p = pretty(pt)
        assert "Point" in p

    def test_fnamedtuple_with_defaults(self):
        Config = fnamedtuple("Config", ["name", "value"], defaults=["default"])
        cfg = Config("test")
        assert cfg.name == "test"
        assert cfg.value == "default"

    def test_fnamedtuple_access(self):
        Vector = fnamedtuple("Vector", ["x", "y", "z"])
        v = Vector(1, 2, 3)
        assert v.x == 1
        assert v.y == 2
        assert v.z == 3
        assert v[0] == 1
        assert v[1] == 2
        assert v[2] == 3


if __name__ == "__main__":
    unittest.main()
