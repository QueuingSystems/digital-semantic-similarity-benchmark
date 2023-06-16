import click
from dss_benchmark.common import print_dataclass
from dss_benchmark.common import parse_arbitrary_arguments
import pandas as pd
from dss_benchmark.methods.anmatveev.train.train import (
    TrainModelParams,
    TrainModelManager
)


@click.group(
    "train", help="Обучение: Обучение неглубоких нейросетей"
)
def trn():
    pass

@trn.command(help="Описание параметров")
def params():
    print_dataclass(TrainModelParams)

@trn.command(
    help="Обучить неглубокую модель", context_settings=dict(ignore_unknown_options=True)
)
@click.option("-mn", "--model_name", required=True, type=str, help="Идентификатор модели", prompt=True)
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
def train_model(model_name, args):
    kwargs = parse_arbitrary_arguments(args)
    params = TrainModelParams(**kwargs)
    print_dataclass(params)
    manager = TrainModelManager(params)


@trn.command(
    help="Обучить неглубокую модель", context_settings=dict(ignore_unknown_options=True)
)
@click.option("-mn", "--model_name", required=True, type=str, help="Идентификатор модели", prompt=True)
# --text тоже обязательный аргумент - путь к файлу
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
def train_model(model_name, args):
    print(model_name)
    kwargs = parse_arbitrary_arguments(args)
    params = TrainModelParams(**kwargs)
    manager = TrainModelManager(params)
    manager.train()


@trn.command(
    help="Предобработать обучающий набор", context_settings=dict(ignore_unknown_options=True)
)
@click.option("-p", "--path", required=True, type=str, help="Путь к файлу с обучающим набором", prompt=True)
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
def preprocess_dataset(path, args):
    manager = TrainModelManager()
    try:
        data = pd.read_json(path)
        new_path = path.replace('.json', '_preprocessed.json')
        manager.preprocess_and_save(data_df=data, text_field='text', path=new_path)
    except FileNotFoundError:
        print('File ' + path + " not exist")
