import cachetools
import click
import pandas as pd
import redis
from dss_benchmark.common import RedisCache, parse_arbitrary_arguments, print_dataclass
from dss_benchmark.experiments import DATASETS, kwm_match, load_dataset
from dss_benchmark.experiments.common import confusion_matrix, f1_score
from dss_benchmark.methods.keyword_matching import (
    KeywordDistanceMatcher,
    KwDistanceMatcherParams,
)

__all__ = ["kwme"]


def _init_cache():
    try:
        cache = RedisCache(client=redis.Redis(host="localhost", password="12345"))
    except Exception:
        print("Cannot initialize redis")
        cache = cachetools.LFUCache(2**16)
    return cache


@click.group(
    "keyword_matching_exp",
    help="Эксперименты: Сопоставление текстов через ключевые слова",
)
def kwme():
    pass


@kwme.command(
    help="Проверить работу на датасете с данными параметрами",
    context_settings=dict(ignore_unknown_options=True),
)
@click.option(
    "-d", "--dataset-name", type=click.Choice(DATASETS), required=True, prompt=True
)
@click.option(
    "-c",
    "--cutoff",
    type=click.IntRange(0, 100, True, True),
    required=True,
    default=50,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def match(dataset_name, cutoff, args):
    cache = _init_cache()
    dataset = load_dataset(dataset_name)
    kwargs = parse_arbitrary_arguments(args)
    params = KwDistanceMatcherParams(**kwargs)
    print_dataclass(params)
    matcher = KeywordDistanceMatcher(params, cache=cache)
    results = kwm_match(matcher, cutoff, dataset, verbose=True)

    df = pd.DataFrame(
        [
            {
                "title_1": result.datum.title_1,
                "title_2": result.datum.title_2,
                "value": result.value,
                "need_match": result.datum.need_match,
                "match": result.match,
            }
            for result in results
        ]
    )
    print(df)

    tp, fp, fn, tn = confusion_matrix(results)
    f1 = f1_score(tp, fp, fn)

    print(f"TP: {str(tp):3s} FP: {str(fp):3s}")
    print(f"FN: {str(fn):3s} TN: {str(tn):3s}")
    print(
        f"F1: {f1:.4f}, precision: {tp / (tp + fp):.4f}, recall: {tp / (tp + fn):.4f}"
    )
