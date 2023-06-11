import time
from multiprocessing.pool import Pool
from multiprocessing.process import current_process
from typing import List

from dss_benchmark.common import tqdm_v, init_cache
from dss_benchmark.methods.keyword_matching import (
    KeywordDistanceMatcher,
    KwDistanceMatcherParams,
)

from .common import Datum, ResultDatum

__all__ = ["kwm_match", "kwm_match_parallel"]


def kwm_match(
    matcher: KeywordDistanceMatcher, cutoff: int, dataset: List[Datum], verbose=False
):
    result: List[ResultDatum] = []
    for datum in tqdm_v(dataset, verbose=verbose):
        value = matcher.match(datum.text_1, datum.text_2)
        result.append(ResultDatum(datum=datum, value=value, match=value >= cutoff))
    return result


def _kwm_match_parallel_init(params, verbose_, cutoff_):
    global matcher, cutoff, verbose
    verbose = verbose_
    cache = init_cache()
    matcher = KeywordDistanceMatcher(params, False, cache)
    cutoff = cutoff_
    if verbose:
        process = current_process()
        print(f"Initialized worker: {process.name} ({process.pid})")


def _kwm_match_task(datum: Datum):
    global matcher, cutoff, verbose
    start_time = time.time()
    value = matcher.match(datum.text_1, datum.text_2)
    end_time = time.time()
    if verbose:
        process = current_process()
        print(f"Finished worker: {process.name} ({process.pid}) in {end_time - start_time:.2f}s")
    return ResultDatum(datum=datum, value=value, match=value >= cutoff)


def kwm_match_parallel(
    params: KwDistanceMatcherParams, cutoff, dataset: List[Datum], verbose=False
):
    result: List[ResultDatum] = []

    with Pool(
        initializer=_kwm_match_parallel_init,
        initargs=(params, verbose, cutoff),
    ) as pool:
        for datum in tqdm_v(
            pool.map(_kwm_match_task, dataset),
            total=len(dataset),
            verbose=verbose,
        ):
            result.append(datum)
    return result
