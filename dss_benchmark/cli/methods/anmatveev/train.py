import click
from dss_benchmark.common import print_dataclass
from dss_benchmark.common import parse_arbitrary_arguments
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
    