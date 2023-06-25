import os

import click
from dss_benchmark.common import init_cache, parse_arbitrary_arguments, print_dataclass
from dss_benchmark.experiments import DATASETS, load_dataset
from dss_benchmark.experiments.tfidf import tfidf_experiment, tfidf_match
from dss_benchmark.methods.tfidf import TfIdfMatcher, TfIdfMatcherParams

from .common import print_results

__all__ = ["tfidfe"]


@click.group(
    "tfidf-exp",
    help="Эксперименты: Сопоставление текстов при помощи метода tf-idf",
)
def tfidfe():
    pass


@tfidfe.command(
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
    params = TfIdfMatcherParams(**kwargs)
    print_dataclass(params)
    matcher = TfIdfMatcher(params, cache=cache)
    results = tfidf_match(matcher, cutoff, dataset, verbose=True)

    print_results(results)


@tfidfe.command(
    help="Провести эксперимент с подбором параметров",
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
        results_folder = f"_output/tfidf_exp_{dataset_name}"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
    tfidf_experiment(dataset, dataset_name, results_folder, True)
