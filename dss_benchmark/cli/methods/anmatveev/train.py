import click
from dss_benchmark.common import print_dataclass
from dss_benchmark.common import parse_arbitrary_arguments

@click.group(
    "train", help="Обучение: Обучение неглубоких нейросетей"
)
def trn():
    pass