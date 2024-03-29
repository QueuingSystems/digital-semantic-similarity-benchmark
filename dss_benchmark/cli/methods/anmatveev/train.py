import click
import pandas as pd
from dss_benchmark.common import parse_arbitrary_arguments, print_dataclass
from dss_benchmark.methods.anmatveev import ANMTrainModelManager, ANMTrainModelParams

__all__ = ["trn"]


@click.group("anm-train", help="Обучение: Обучение неглубоких нейросетей")
def trn():
    pass


@trn.command(help="Описание параметров")
def params():
    print_dataclass(ANMTrainModelParams)


@trn.command(
    help="Обучить неглубокую модель", context_settings=dict(ignore_unknown_options=True)
)
@click.option(
    "-mn",
    "--model_name",
    required=True,
    type=str,
    help="Идентификатор модели",
    prompt=True,
)
@click.option(
    "-mp",
    "--model_path",
    required=True,
    type=str,
    help="Путь к будущей модели",
    prompt=True,
)
# --text тоже обязательный аргумент - путь к файлу
@click.option("-t", "--text", required=True, type=str, help="Путь к файлу", prompt=True)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def train_model(model_name, model_path, text, args):
    kwargs = parse_arbitrary_arguments(args)
    params = ANMTrainModelParams(**kwargs)
    params.texts = text
    manager = ANMTrainModelManager(params)
    manager.train(model=model_name, model_path=model_path)


@trn.command(
    help="Предобработать обучающий набор",
    context_settings=dict(ignore_unknown_options=True),
)
@click.option(
    "-p",
    "--path",
    required=True,
    type=str,
    help="Путь к файлу с обучающим набором",
    prompt=True,
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def preprocess_dataset(path, args):
    manager = ANMTrainModelManager()
    try:
        data = pd.read_json(path)
        new_path = path.replace(".json", "_preprocessed.json")
        manager.preprocess_and_save(data_df=data, text_field="text", path=new_path)
    except FileNotFoundError:
        print("File " + path + " not exist")
