import click
from dss_benchmark.common import print_dataclass
from dss_benchmark.methods.keyword_matching import (
    KeywordDistanceMatcher,
    KwDistanceMatcherParams,
)

__all__ = ["kwm"]


@click.group(help="Методы. Сопоставление текстов через ключевые слова")
def kwm():
    pass


@kwm.command(help="Описание параметров")
def help():
    print_dataclass(KwDistanceMatcherParams)


@kwm.command(help='Сматчить 2 текста')
def match():
    pass
