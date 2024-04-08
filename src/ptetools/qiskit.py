from typing import Any

import numpy as np


def counts2dense(c: dict[str, Any], number_of_bits: int) -> np.ndarray:
    """Convert dictionary with fractions or counts to a dense array"""
    d = np.zeros(2**number_of_bits, dtype=np.array(sum(c.values())).dtype)
    for k, v in c.items():
        idx = int(k.replace(" ", ""), base=2)
        d[idx] = v
    return d
