import click
from dss_benchmark.common import parse_arbitrary_arguments, print_dataclass
from dss_benchmark.methods.tfidf import TfIdfMatcher, TfIdfMatcherParams

__all__ = ["tfidf_"]


@click.group("tfidf", help="Методы: Сопоставление текстов при помощи метода tfidf")
def tfidf_():
    pass


@tfidf_.command(help="Описание параметров")
def params():
    print_dataclass(TfIdfMatcherParams)


@tfidf_.command(
    help="Сматчить 2 текста", context_settings=dict(ignore_unknown_options=True)
)
@click.option("-t1", "--text1", required=True, type=str, help="Текст 1", prompt=True)
@click.option("-t2", "--text2", required=True, type=str, help="Текст 2", prompt=True)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def match(text1, text2, args):
    kwargs = parse_arbitrary_arguments(args)
    params = TfIdfMatcherParams(**kwargs)
    print_dataclass(params)
    matcher = TfIdfMatcher(params)
    result = matcher.match(text1, text2)
    print(f"\nСходство: {result}")
