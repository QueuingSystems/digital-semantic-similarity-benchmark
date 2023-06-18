import click
from dss_benchmark.common import print_dataclass
from dss_benchmark.common import parse_arbitrary_arguments
import pandas as pd
from dss_benchmark.methods.anmatveev.match import (
    MatchManager
)


@click.group(
    "match", help="Сопоставление и исследование"
)
def mch():
    pass


@mch.command(
    help="Максимизировать F1-score с подбором оптимального cutoff", context_settings=dict(ignore_unknown_options=True)
)
@click.option("-mp", "--model_path", required=True, type=str, help="Путь к модели", prompt=True)
@click.option("-t", "--texts", required=True, type=str, help="Путь к бенчмарку", prompt=True)
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
def max_f1(model_path, texts, args=None):
    manager = MatchManager(model_path)
    benchmark = pd.read_json(texts)
    res = manager.max_f1(benchmark['text_rp'],
                         benchmark['text_proj'],
                         benchmark,
                         model_path)
    print(res)
    return res


@mch.command(
    help="Построить ROC-кривую с вычислением площади AUC", context_settings=dict(ignore_unknown_options=True)
)
@click.option("-mp", "--model_path", required=True, type=str, help="Путь к модели", prompt=True)
@click.option("-t", "--texts", required=True, type=str, help="Путь к бенчмарку", prompt=True)
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
def roc_auc(model_path, texts, args):
    manager = MatchManager(model_path)
    benchmark = pd.read_json(texts)
    res = manager.roc_auc(benchmark['text_rp'],
                          benchmark['text_proj'],
                          benchmark,
                          model_path)
    print(res)
    return res
