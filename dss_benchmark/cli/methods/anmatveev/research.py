import os

import click
from dss_benchmark.methods.anmatveev.train import *
from dss_benchmark.methods.anmatveev.match import *
from dss_benchmark.methods.anmatveev.plot import *
from dss_benchmark.methods.anmatveev.params_parser import *
import pandas as pd


@click.group(
    "research", help="Сопоставление и исследование"
)
def rsch():
    pass


@rsch.command(
    help="Обучить все возможные комбинации моделей",
    context_settings=dict(ignore_unknown_options=True)
)
@click.option("-fp", "--file_path", required=True, type=str, help="Путь к файлу с параметрами", prompt=True)
@click.option("-tr", "--train_text", required=True, type=str, help="Обучающий набор", prompt=True)
@click.option("-mp", "--models_path", required=True, type=str, help="Путь к моделям", prompt=True)
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
# pretrain models before its comparing with splitting into groups
def train_cascade(file_path, train_text, models_path, args):
    kwargs = parse_arbitrary_arguments(args)
    params = TrainModelParams(**kwargs)
    params.texts = train_text
    os.makedirs(models_path, exist_ok=True)
    parser = ParamsParser(file_path)
    groups = parser.read(split_into_groups=True)
    for group in groups:
        subdir = os.path.join(models_path, group)
        os.makedirs(subdir, exist_ok=True)
        i = 1
        for case in groups[group]:
            params.window = case[1][1]
            params.epochs = case[1][2]
            params.sg = case[1][3]
            params.min_count = case[1][4]
            params.vector_size = case[1][5]
            # in fastText maybe sth else
            trainManager = TrainModelManager(params)
            trainManager.train(case[1][0], model_path=os.path.join(subdir, str(i) + '-' + case[0]))
            i += 1
            print(case[0])


@rsch.command(
    help="Подбор оптимальных параметров для максимизации F1-score",
    context_settings=dict(ignore_unknown_options=True)
)
@click.option("-mp", "--models_path", required=True, type=str, help="Путь к папке с моделями", prompt=True)
@click.option("-bt", "--benchmark_text", required=True, type=str, help="Путь к бенчмарку", prompt=True)
@click.option("-t1", "--text1", required=True, type=str, help="Поле 1", prompt=True)
@click.option("-t2", "--text2", required=True, type=str, help="Поле 2", prompt=True)
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
# cases in a file will be splitted into groups with a space
def get_best_params_f1(models_path, benchmark_text, text1, text2, args):
    kwargs = parse_arbitrary_arguments(args)
    params = TrainModelParams(**kwargs)
    benchmark_text = pd.read_json(benchmark_text)
    for group, dirs, files in sorted(os.walk(models_path))[1:]:
        plotManager = PlotManager()
        imname = f"Maximization-F1-score-{files[0].split('-')[1]}"
        plotManager.init_plot(title=imname,
                              xlabel="Cutoff",
                              ylabel="F1-score",
                              figsize=(7, 6))

        for file in sorted(files):
            model_path = os.path.join(group, file)
            case = ParamsParser().read_one(file)
            params.window = case[1]
            params.epochs = case[2]
            params.sg = case[3]
            params.min_count = case[4]
            params.vector_size = case[5]
            matchManager = MatchManager(model_path, params)
            res = matchManager.max_f1(benchmark_text[text1], benchmark_text[text2], benchmark_text,
                                      model_path)
            res["window"] = params.window
            res["epochs"] = params.epochs
            res["sg"] = params.sg
            res["min_count"] = params.min_count
            res["vector_size"] = params.vector_size
            plotManager.add_plot(res, plot_type="F1-score")
        plotManager.save(group.split('/')[-1] + '-'+ imname + ".png")


