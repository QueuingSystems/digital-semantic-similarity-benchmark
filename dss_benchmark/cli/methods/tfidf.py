import click
from dss_benchmark.common import print_dataclass
from dss_benchmark.common import parse_arbitrary_arguments
from dss_benchmark.methods.tfidf_transformers.tfidf import (
    TfIdf,
    TfIdfParams
)

__all__ = ["tfidf"]


@click.group(
    "tfidf", help="Методы: Сопоставление текстов при помощи метода tfidf"
)
def tfidf():
    pass


@tfidf.command(help="Описание параметров")
def params():
    print_dataclass(TfIdfParams)


@tfidf.command(
    help="Сматчить 2 текста", context_settings=dict(ignore_unknown_options=True)
)
@click.option("-t1", "--text1", required=True, type=str, help="Текст 1", prompt=True)
@click.option("-t2", "--text2", required=True, type=str, help="Текст 1", prompt=True)
@click.argument(
    "args", nargs=-1, type=click.UNPROCESSED
)
def match(text1, text2, args):
    kwargs = parse_arbitrary_arguments(args)
    params = TfIdfParams(**kwargs)
    print_dataclass(params)
    matcher = TfIdf(params)
    result = matcher.match(text1, text2)
    print(f'\nСходство: {result}')
