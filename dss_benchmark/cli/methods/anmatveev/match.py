from pathlib import Path

import click
import pandas as pd
from dss_benchmark.common import parse_arbitrary_arguments
from dss_benchmark.methods.anmatveev import (
    ANMMatchManager,
    ANMPlotManager,
    ANMTrainModelParams,
    ParamsParser,
)

__all__ = ["mch"]


@click.group("anm-match", help="Сопоставление и исследование")
def mch():
    pass


@mch.command(
    help="Максимизировать F1-score с подбором оптимального cutoff",
    context_settings=dict(ignore_unknown_options=True),
)
@click.option(
    "-mp", "--model_path", required=True, type=str, help="Путь к модели", prompt=True
)
@click.option(
    "-mt",
    "--model_type",
    required=True,
    type=click.Choice(["gensim", "transformer"]),
    help="Тип модели",
    prompt=True,
)
@click.option(
    "-t", "--texts", required=True, type=str, help="Путь к бенчмарку", prompt=True
)
@click.option("-t1", "--text1", required=True, type=str, help="Поле 1", prompt=True)
@click.option("-t2", "--text2", required=True, type=str, help="Поле 2", prompt=True)
@click.option(
    "-imp",
    "--im_prefix",
    required=True,
    type=str,
    help="Префикс названия картинки",
    prompt=True,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def max_f1(model_path, model_type, texts, text1, text2, im_prefix, args):
    benchmark = None
    res = None
    if ".json" in texts:
        benchmark = pd.read_json(texts)
    elif ".csv" in texts:
        benchmark = pd.read_csv(texts)
    if model_type == "gensim":
        kwargs = parse_arbitrary_arguments(args)
        params = ANMTrainModelParams(**kwargs)
        manager = ANMMatchManager(model_path, params)
        case = ParamsParser().read_one(model_path)
        params.sg = case[1]
        params.window = case[2]
        params.epochs = case[3]
        params.min_count = case[4]
        params.vector_size = case[5]
        if "fastText".lower() in model_path.lower():
            params.min_n = case[6]
            params.max_n = case[7]
        print(case)
        imname = f"Maximization-F1-score-{case[0]}"
        plotManager = ANMPlotManager()
        plotManager.init_plot(
            title=imname,
            xlabel="Cutoff",
            ylabel="F1-score",
            plot_type="F1-score",
            model=case[0],
            figsize=(7, 6),
        )
        res = manager.max_f1(benchmark[text1], benchmark[text2], benchmark, model_path)
        res["window"] = params.window
        res["epochs"] = params.epochs
        res["sg"] = params.sg
        res["min_count"] = params.min_count
        res["vector_size"] = params.vector_size
        res["min_n"] = params.min_n
        res["max_n"] = params.max_n
        plotManager.add_plot(res)
        plotManager.save(im_prefix + imname)
    elif model_type == "transformer":
        model = Path(model_path).stem
        if model.lower().startswith("paraphrase-multilingual"):
            model = "multilingual"
        imname = f"Maximization-F1-score-{model}"
        plotManager = ANMPlotManager()
        plotManager.init_plot(
            title=imname,
            xlabel="Cutoff",
            ylabel="F1-score",
            plot_type="F1-score",
            model=model,
            figsize=(7, 6),
        )
        manager = ANMMatchManager(model_path)
        res = manager.max_f1(benchmark[text1], benchmark[text2], benchmark, model_path)
        plotManager.add_plot(res)
        plotManager.save(im_prefix + imname, legend_fs=14)
    return res


@mch.command(
    help="Построить ROC-кривую с вычислением площади AUC",
    context_settings=dict(ignore_unknown_options=True),
)
@click.option(
    "-mp", "--model_path", required=True, type=str, help="Путь к модели", prompt=True
)
@click.option(
    "-t", "--texts", required=True, type=str, help="Путь к бенчмарку", prompt=True
)
@click.option("-t1", "--text1", required=True, type=str, help="Поле 1", prompt=True)
@click.option("-t2", "--text2", required=True, type=str, help="Поле 2", prompt=True)
@click.option(
    "-mt",
    "--model_type",
    required=True,
    type=click.Choice(["gensim", "transformer"]),
    help="Тип модели",
    prompt=True,
)
@click.option(
    "-imp",
    "--im_prefix",
    required=True,
    type=str,
    help="Префикс названия картинки",
    prompt=True,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def roc_auc(model_path, texts, text1, text2, model_type, im_prefix, args):
    benchmark = None
    if ".json" in texts:
        benchmark = pd.read_json(texts)
    elif ".csv" in texts:
        benchmark = pd.read_csv(texts)
    res = None
    if model_type == "gensim":
        kwargs = parse_arbitrary_arguments(args)
        params = ANMTrainModelParams(**kwargs)
        manager = ANMMatchManager(model_path, params=params)
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
        plotManager = ANMPlotManager()
        imname = f"ROC-AUC-{case[0]}"
        plotManager.init_plot(
            title=imname,
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            model=case[0],
            plot_type="ROC-AUC",
            figsize=(7, 6),
        )

        res = manager.roc_auc(benchmark[text1], benchmark[text2], benchmark, model_path)
        res["window"] = params.window
        res["epochs"] = params.epochs
        res["sg"] = params.sg
        res["min_count"] = params.min_count
        res["vector_size"] = params.vector_size
        res["min_n"] = params.min_n
        res["max_n"] = params.max_n
        plotManager.add_plot(res)
        plotManager.save(im_prefix + imname)
    elif model_type == "transformer":
        model = Path(model_path).stem
        if model.lower().startswith("paraphrase-multilingual"):
            model = "multilingual"
        imname = f"ROC-AUC-{model}"
        plotManager = ANMPlotManager()
        plotManager.init_plot(
            title=imname,
            xlabel="False Positive Rate",
            ylabel="True Positive Rate",
            model=model,
            plot_type="ROC-AUC",
            figsize=(7, 6),
        )
        manager = ANMMatchManager(model_path)
        res = manager.roc_auc(benchmark[text1], benchmark[text2], benchmark, model_path)
        plotManager.add_plot(res)
        plotManager.save(im_prefix + imname, legend_fs=14)
    return res
