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
def train_cascade(file_path, train_text, models_path, args):
    kwargs = parse_arbitrary_arguments(args)
    params = TrainModelParams(**kwargs)
    params.texts = train_text
    os.makedirs(models_path, exist_ok=True)
    print(models_path)
    parser = ParamsParser(file_path)
    i = 0
    for case in parser.read():
        i+=1
        params.window = case[1][1]
        params.epochs = case[1][2]
        params.sg = case[1][3]
        params.min_count = case[1][4]
        params.vector_size = case[1][5]
        print(params)
        trainManager = TrainModelManager(params)
        print(case[0], i)
        model_path = trainManager.train(case[1][0], model_path=os.path.join(models_path, str(i) +"-"+ case[0]))

