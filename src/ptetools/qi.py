import datetime
from typing import Any

try:
    from quantuminspire.credentials import get_token_authentication, load_account
    from quantuminspire.qiskit import QI
except ImportError:
    import warnings

    warnings.warn("could not import quantuminspire. install it via pip install quanuminspire")


def starmon5_backend(backend_name: str = "Starmon-5", number_of_shots: int | None = None) -> Any:  # pragma: no cover
    """Connect to QI Starmon backend

    Args:
        backend: Name of the backend can be "Starmon-5" or "QX single-node simulator"
    Returns:
        Tuple with backend name and backend
    """
    authentication = get_token_authentication(load_account())

    QI.set_authentication(authentication)
    qi_backend = QI.get_backend(backend_name)
    if number_of_shots:
        qi_backend.options.shots = number_of_shots  # for starmon-5 overhead is in compilation
    return qi_backend


# %%
def qi_counts2qiskit(counts: dict[str, int], num_bits: int) -> dict[str, int]:
    """Convert measurement histogram from qi 1.0 to qiskit convention"""
    fmt = f"{{:0{num_bits}b}}"

    def convert(i):
        v = int(i, 16)
        return fmt.format(v)

    return {convert(k): v for k, v in counts.items()}


if __name__ == "__main__":
    from qiskit.result import marginal_counts

    mm = [{"0x0": 8182, "0x4": 10}, {"0x0": 4137, "0x4": 4055}, {"0x0": 263, "0x4": 7929}]
    mmq = [qi_counts2qiskit(m, 5) for m in mm]
    print(mmq)
    print(marginal_counts(mmq[1], [2]))


# %%
def report_qi_status():  # pragma: no cover
    from quantuminspire.api import QuantumInspireAPI
    from quantuminspire.credentials import get_token_authentication, load_account
    from rich import print as rprint

    QI_URL = r"https://api.quantum-inspire.com"

    token = load_account()
    authentication = get_token_authentication(token)
    qi_api = QuantumInspireAPI(QI_URL, authentication)

    bb = qi_api.get_backend_types()

    ts = datetime.datetime.now().isoformat()
    print(f"{ts}: QI backends:")
    for backend in bb:
        name = backend["name"]
        status = backend["status"]
        if status == "OFFLINE":
            rprint(f"  backend {name}: [red]{status}[/red] ")
        else:
            print(f"  backend {name}: {status} ")
    print("to run a test job, use run_qi_test_job")


if __name__ == "__main__":
    report_qi_status()
