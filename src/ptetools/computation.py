import itertools
from collections.abc import Callable, Sequence
from typing import Any

from joblib import Parallel, delayed
from tqdm import tqdm

from ptetools.tools import make_blocks  # noqa: F401


def parallel_execute(
    method: Callable,
    data: Sequence[Any],
    seed: None | int = None,
    *,
    n_jobs: int = 5,
    block_size: int | None = None,
    progress_bar: str = "block",
    **kwargs,
):
    """Parallel execution of computationally intensive methods"""
    data = tuple(data)
    number_of_datapoints = len(data)
    if block_size is None:
        block_size = max(number_of_datapoints // 5, 1)
    blocks = make_blocks(number_of_datapoints, block_size)

    def execution_method(block):
        return [method(**data[i]) for i in range(*block)]

    parjob = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(execution_method)(block=block) for block in (blocks)
    )
    if progress_bar:
        results = list(tqdm(parjob, total=len(blocks), desc=progress_bar))
    else:
        results = list(parjob)

    return list(itertools.chain(*results))
