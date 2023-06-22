import click
from dss_benchmark.common import print_dataclass
from dss_benchmark.common import parse_arbitrary_arguments
import pandas as pd

from dss_benchmark.methods.anmatveev import TrainModelParams
from dss_benchmark.methods.anmatveev.match import (
    MatchManager
)
from dss_benchmark.methods.anmatveev.params_parser import ParamsParser
from dss_benchmark.methods.anmatveev.plot import PlotManager


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
@click.option("-t1", "--text1", required=True, type=str, help="Поле 1", prompt=True)
@click.option("-t2", "--text2", required=True, type=str, help="Поле 2", prompt=True)
@click.option("-imp", "--im_prefix", required=True, type=str, help="Префикс названия картинки", prompt=True)
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
def max_f1(model_path, texts, text1, text2, im_prefix, args):
    kwargs = parse_arbitrary_arguments(args)
    params = TrainModelParams(**kwargs)
    manager = MatchManager(model_path, params)
    case = ParamsParser().read_one(model_path)
    params.window = case[1]
    params.epochs = case[2]
    params.sg = case[3]
    params.min_count = case[4]
    params.vector_size = case[5]
    if "fastText".lower() in model_path.lower():
        params.min_n = case[6]
        params.max_n = case[7]
    print(case)
    imname = f"Maximization-F1-score-{case[0]}"
    plotManager = PlotManager()
    plotManager.init_plot(title=imname,
                          xlabel="Cutoff",
                          ylabel="F1-score",
                          plot_type="F1-score",
                          model=case[0],
                          figsize=(7, 6))
    benchmark = pd.read_json(texts)
    res = manager.max_f1(benchmark[text1],
                         benchmark[text2],
                         benchmark,
                         model_path)
    res["window"] = params.window
    res["epochs"] = params.epochs
    res["sg"] = params.sg
    res["min_count"] = params.min_count
    res["vector_size"] = params.vector_size
    res["min_n-max_n"] = (params.min_n, params.max_n)
    plotManager.add_plot(res)
    plotManager.save(im_prefix + imname)
    return res


@mch.command(
    help="Построить ROC-кривую с вычислением площади AUC", context_settings=dict(ignore_unknown_options=True)
)
@click.option("-mp", "--model_path", required=True, type=str, help="Путь к модели", prompt=True)
@click.option("-t", "--texts", required=True, type=str, help="Путь к бенчмарку", prompt=True)
@click.option("-t1", "--text1", required=True, type=str, help="Поле 1", prompt=True)
@click.option("-t2", "--text2", required=True, type=str, help="Поле 2", prompt=True)
@click.option("-imp", "--im_prefix", required=True, type=str, help="Префикс названия картинки", prompt=True)
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
def roc_auc(model_path, texts,  text1, text2, im_prefix, args):
    kwargs = parse_arbitrary_arguments(args)
    params = TrainModelParams(**kwargs)
    manager = MatchManager(model_path, params=params)
    benchmark = pd.read_json(texts)
    res = manager.roc_auc(benchmark[text1],
                          benchmark[text2],
                          benchmark,
                          model_path)

    case = ParamsParser().read_one(model_path)
    params.window = case[1]
    params.epochs = case[2]
    params.sg = case[3]
    params.min_count = case[4]
    params.vector_size = case[5]
    if "fastText".lower() in model_path.lower():
        params.min_n = case[6]
        params.max_n = case[7]
    print(case)
    plotManager = PlotManager()
    imname = f"ROC-AUC-{case[0]}"
    plotManager.init_plot(title=imname,
                          xlabel="False Positive Rate",
                          ylabel="True Positive Rate",
                          model=case[0],
                          plot_type="ROC-AUC",
                          figsize=(7, 6))
    benchmark = pd.read_json(texts)
    res = manager.roc_auc(benchmark[text1],
                         benchmark[text2],
                         benchmark,
                         model_path)
    res["window"] = params.window
    res["epochs"] = params.epochs
    res["sg"] = params.sg
    res["min_count"] = params.min_count
    res["vector_size"] = params.vector_size
    res["min_n-max_n"] = (params.min_n, params.max_n)
    plotManager.add_plot(res)
    print(res)
    plotManager.save(im_prefix + imname)
    return res
