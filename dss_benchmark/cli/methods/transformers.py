import click
from dss_benchmark.common import print_dataclass
from dss_benchmark.common import parse_arbitrary_arguments
from dss_benchmark.methods.tfidf_transformers.transormers import (
    Transformers,
    TransformersParams,
)

__all__ = ["tranformes"]


@click.group(
    "tranformes", help="Методы: Сопоставление текстов при помощи трансформеров"
)
def tranformes():
    pass


@tranformes.command(help="Описание параметров")
def params():
    print_dataclass(TransformersParams)


@tranformes.command(
    help="Сматчить 2 текста", context_settings=dict(ignore_unknown_options=True)
)
@click.option("-t1", "--text1", required=True, type=str, help="Текст 1", prompt=True)
@click.option("-t2", "--text2", required=True, type=str, help="Текст 1", prompt=True)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def match(text1, text2, args):
    kwargs = parse_arbitrary_arguments(args)
    params = TransformersParams(**kwargs)
    print_dataclass(params)
    matcher = Transformers(params)
    result = matcher.match(text1, text2)
    print(f"\nСходство: {result}")
