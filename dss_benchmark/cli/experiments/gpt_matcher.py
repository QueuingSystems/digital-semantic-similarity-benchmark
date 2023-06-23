import random

import click
import pandas as pd
from dss_benchmark.common import init_cache, parse_arbitrary_arguments, print_dataclass
from dss_benchmark.experiments import (
    DATASETS,
    confusion_matrix,
    f1_score,
    gpt_match,
    load_dataset,
    process_auprc,
    process_f1_score,
)
from dss_benchmark.experiments.common import process_roc_auc
from dss_benchmark.methods.chatgpt import GPTMatcher, GPTMatcherParams

__all__ = ["gptme"]


@click.group("gpt-matching-exp", help="Эксперименты: сопоставление через промпты к GPT")
def gptme():
    pass


@gptme.command(
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
@click.option("-s", "--shot", type=click.INT, multiple=True, help="Индекс образца")
@click.option(
    "-sv", "--shot-value", type=click.INT, multiple=True, help="Значение образца"
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def match(dataset_name, cutoff, shot, shot_value, args):
    cache = init_cache()
    dataset = load_dataset(dataset_name)
    # random.Random(42).shuffle(dataset)
    kwargs = parse_arbitrary_arguments(args)
    params = GPTMatcherParams(**kwargs)
    print_dataclass(params)
    matcher = GPTMatcher(params, cache=cache, verbose=False)

    assert len(shot) == len(shot_value)

    for idx, value in zip(shot, shot_value):
        datum = dataset[idx]
        matcher.add_shot(datum.text_1, datum.text_2, int(value / 10), tokens=2000)

    results = gpt_match(matcher, cutoff, dataset, verbose=True)

    f1, cutoff = process_f1_score(results)
    print(f"Optimal cutoff: {cutoff:.4f}, F1: {f1:.4f}")

    for result in results:
        result.match = result.value >= cutoff

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

    precision, recall = 0, 0
    if tp + fp > 0:
        precision = tp / (tp + fp)
    if tp + fn > 0:
        recall = tp / (tp + fn)

    print(f"TP: {str(tp):3s} FP: {str(fp):3s}")
    print(f"FN: {str(fn):3s} TN: {str(tn):3s}")
    print(f"F1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")

    _, _, auprc, auprc_cutoff, auprc_f1 = process_auprc(dataset, results)
    print(f"AUPRC: {auprc:.4f}, cutoff: {auprc_cutoff:.4f}, F1:{auprc_f1:.4f}")

    _, _, auc_cutoff, auc = process_roc_auc(dataset, results)
    print(f"AUC: {auc:.4f}, cutoff: {auc_cutoff:.4f}")
