import dataclasses
import os

import click
import pandas as pd
from dss_benchmark.common import (
    append_to_csv,
    init_cache,
    parse_arbitrary_arguments,
    print_dataclass,
)
from dss_benchmark.experiments import (
    DATASETS,
    confusion_matrix,
    f1_score,
    kwm_match,
    kwm_match_parallel,
    load_dataset,
    process_f1_score,
    process_roc_auc,
)
from dss_benchmark.experiments.common import process_auprc
from dss_benchmark.experiments.keyword_matching import kwm_experiment
from dss_benchmark.methods.keyword_matching import (
    KeywordDistanceMatcher,
    KwDistanceMatcherParams,
)

__all__ = ["kwme"]


@click.group(
    "keyword-matching-exp",
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
    cache = init_cache()
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


@kwme.command(
    help="Как match, но с AUC вместо cutoff и сохранить в CSV",
    context_settings=dict(ignore_unknown_options=True),
)
@click.option(
    "-d", "--dataset-name", type=click.Choice(DATASETS), required=True, prompt=True
)
@click.option(
    "-c",
    "--csv-path",
    type=click.Path(dir_okay=False),
    required=True,
    default="_output/kwm.csv",
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def match_auc(dataset_name, csv_path, args):
    dataset = load_dataset(dataset_name)
    kwargs = parse_arbitrary_arguments(args)
    params = KwDistanceMatcherParams(**kwargs)
    print_dataclass(params)
    results = kwm_match_parallel(params, 0, dataset, verbose=True)
    # cache = init_cache()
    # matcher = KeywordDistanceMatcher(params, cache=cache)
    # results = kwm_match(matcher, 0, dataset, verbose=True)

    _, _, auc_cutoff, auc = process_roc_auc(dataset, results)
    for r in results:
        r.match = r.value >= auc_cutoff
    tp, fp, fn, tn = confusion_matrix(results)
    f1 = f1_score(tp, fp, fn)
    print(f"AUC: {auc:.4f} (cutoff: {auc_cutoff:.4f}, F1: {f1:.4f})")

    _, _, auprc, auprc_cutoff, auprc_f1 = process_auprc(dataset, results)
    print(f'AUPRC: {auprc:.4f} (cutoff: {auprc_cutoff:.4f}, F1: {auprc_f1:.4f})')

    f1, cutoff = process_f1_score(results)
    for r in results:
        r.match = r.value >= cutoff

    tp, fp, fn, tn = confusion_matrix(results)
    f1 = f1_score(tp, fp, fn)

    print(f"F1: {f1:.4f} (cutoff: {cutoff:.4f})")

    print(f"TP: {str(tp):3s} FP: {str(fp):3s}")
    print(f"FN: {str(fn):3s} TN: {str(tn):3s}")
    print(f"precision: {tp / (tp + fp):.4f}, recall: {tp / (tp + fn):.4f}")
    append_to_csv(
        csv_path,
        {
            "dataset": dataset_name,
            **dataclasses.asdict(params),
            "auc": auc,
            "auc_cutoff": auc_cutoff,
            "auprc": auprc,
            "auprc_cutoff": auprc_cutoff,
            "auprc_f1": auprc_f1,
            "f1": f1,
            "cutoff": cutoff,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "precision": tp / (tp + fp),
            "recall": tp / (tp + fn),
        },
    )


@kwme.command(
    help="Провести эксперимент с подбором параметров",
    context_settings=dict(ignore_unknown_options=True),
)
@click.option(
    "-d", "--dataset-name", type=click.Choice(DATASETS), required=True, prompt=True
)
@click.option(
    "-r",
    "--results-folder",
    type=click.Path(dir_okay=True),
)
def match_exp(dataset_name, results_folder):
    dataset = load_dataset(dataset_name)
    if results_folder is None:
        results_folder = f"_output/kwm_exp_{dataset_name}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    kwm_experiment(dataset, dataset_name, results_folder, True)
