import io
import unittest
from contextlib import redirect_stdout

from ptetools.qi import qi_counts2qiskit, report_qi_status


class TestQi(unittest.TestCase):
    def test_report_qi_status(self):
        return  # test not enabled
        with redirect_stdout(io.StringIO()) as s:
            report_qi_status()
        assert "QI backends" in s.getvalue()


def test_qi_counts2qiskit():
    mm = [{"0x0": 8182, "0x4": 10}, {"0x0": 4137, "0x4": 4055}, {"0x0": 263, "0x4": 7929}]
    mmq = [qi_counts2qiskit(m, 5) for m in mm]
    assert mmq == [{"00000": 8182, "00100": 10}, {"00000": 4137, "00100": 4055}, {"00000": 263, "00100": 7929}]


if __name__ == "__main__":
    unittest.main()
