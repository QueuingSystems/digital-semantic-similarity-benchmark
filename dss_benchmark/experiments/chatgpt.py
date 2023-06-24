from typing import List

from dss_benchmark.common import init_cache, tqdm_v
from dss_benchmark.experiments.common import Datum, ResultDatum
from dss_benchmark.methods.chatgpt import GPTMatcher, GPTMatcherParams

from .keyword_matching import _kwm_get_big_dataset, _kwm_measure_time_on_dataset

__all__ = ["gpt_match", "gpt_measure_timings"]


def gpt_match(matcher: GPTMatcher, cutoff: int, dataset: List[Datum], verbose=False):
    result: List[ResultDatum] = []
    for datum in tqdm_v(dataset, verbose=verbose):
        value = matcher.match(datum.text_1, datum.text_2)
        result.append(ResultDatum(datum=datum, value=value, match=value >= cutoff))
    return result


def gpt_measure_timings(verbose=False):
    big_dataset = _kwm_get_big_dataset(verbose=verbose, one_student=True)
    default_params = GPTMatcherParams()
    cache = init_cache("memory")
    matcher = GPTMatcher(default_params, False, cache)

    df_no_cache = _kwm_measure_time_on_dataset(matcher, big_dataset, verbose)
    df_no_cache.to_csv("_output/gpt_times_no_cache.csv", index=False)

    df_cached = _kwm_measure_time_on_dataset(matcher, big_dataset, verbose)
    df_cached.to_csv("_output/gpt_times_cached.csv", index=False)

    big_dataset[
        0
    ].text_2 += ". Этот текст был изменен, чтобы проверить скорость пересчёта"
    df_changed_vacancy = _kwm_measure_time_on_dataset(matcher, big_dataset, verbose)
    df_changed_vacancy.to_csv(
        "_output/gpt_times_no_cache_changed_vacancy.csv", index=False
    )

    for datum in big_dataset:
        datum.text_1 += ". Этот текст был изменен, чтобы проверить скорость пересчёта"
    df_changed_resume = _kwm_measure_time_on_dataset(matcher, big_dataset, verbose)
    df_changed_resume.to_csv(
        "_output/gpt_times_no_cache_changed_resume.csv", index=False
    )

    big_dataset_swapped = []
    for datum in big_dataset:
        big_dataset_swapped.append(
            Datum(
                text_1=datum.text_2,
                text_2=datum.text_1,
                title_1=datum.title_2,
                title_2=datum.title_1,
                need_match=datum.need_match,
            )
        )
    df_swapped = _kwm_measure_time_on_dataset(matcher, big_dataset_swapped, verbose)
    df_swapped.to_csv("_output/gpt_times_no_cache_swapped.csv", index=False)
