import itertools
from collections.abc import Callable, Sequence

from joblib import Parallel, delayed

from ptetools.tools import tqdm


def make_blocks(size: int, block_size: int) -> list[tuple[int, int]]:
    number_of_blocks = (size + block_size - 1) // block_size
    blocks = [(ii * block_size, min(size, (ii + 1) * block_size)) for ii in range(number_of_blocks)]
    return blocks


def parallel_execute(
    method: Callable,
    data: Sequence[any],
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

    def execution_method(block, **kwargs):
        return [method(**data[i]) for i in range(*block)]

    parjob = Parallel(n_jobs=5, return_as="generator")(delayed(execution_method)(block=block) for block in (blocks))
    if progress_bar:
        results = list(tqdm(parjob, total=len(blocks), desc=progress_bar))
    else:
        results = list(parjob)

    return list(itertools.chain(*results))
