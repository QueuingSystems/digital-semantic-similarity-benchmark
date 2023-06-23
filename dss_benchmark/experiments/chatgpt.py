from typing import List

from dss_benchmark.common import tqdm_v
from dss_benchmark.experiments.common import Datum, ResultDatum
from dss_benchmark.methods.chatgpt import GPTMatcher

__all__ = ["gpt_match"]


def gpt_match(matcher: GPTMatcher, cutoff: int, dataset: List[Datum], verbose=False):
    result: List[ResultDatum] = []
    for datum in tqdm_v(dataset, verbose=verbose):
        value = matcher.match(datum.text_1, datum.text_2)
        result.append(ResultDatum(datum=datum, value=value, match=value >= cutoff))
    return result
