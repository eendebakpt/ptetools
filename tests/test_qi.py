import io
import unittest
from contextlib import redirect_stdout

from ptetools.qi import report_qi_status


class TestQi(unittest.TestCase):
    def test_report_qi_status(self):
        with redirect_stdout(io.StringIO()) as s:
            report_qi_status()
        assert "QI backends" in s.getvalue()


if __name__ == "__main__":
    unittest.main()
