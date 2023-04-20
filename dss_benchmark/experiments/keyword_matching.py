from typing import List

from dss_benchmark.common import tqdm_v
from dss_benchmark.methods.keyword_matching import KeywordDistanceMatcher

from .common import Datum, ResultDatum

__all__ = ["kwm_match"]


def kwm_match(
    matcher: KeywordDistanceMatcher, cutoff: int, dataset: List[Datum], verbose=False
):
    result: List[ResultDatum] = []
    for datum in tqdm_v(dataset, verbose=verbose):
        value = matcher.match(datum.text_1, datum.text_2)
        result.append(ResultDatum(datum=datum, value=value, match=value >= cutoff))
    return result
