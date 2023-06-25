import click

from dss_benchmark.common import (init_cache, parse_arbitrary_arguments,
                                  print_dataclass)
from dss_benchmark.methods.chatgpt import GPTMatcher, GPTMatcherParams

__all__ = ["gpt"]


@click.group("gpt-matching", help="Методы: Сопоставление текстов через GPT-3.5")
def gpt():
    pass


@gpt.command(help="Описание параметров")
def params():
    print_dataclass(GPTMatcherParams)


@gpt.command(
    help="Сматчить 2 текста", context_settings=dict(ignore_unknown_options=True)
)
@click.option("-t1", "--text1", required=True, type=str, help="Текст 1", prompt=True)
@click.option("-t2", "--text2", required=True, type=str, help="Текст 1", prompt=True)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def match(text1, text2, args):
    kwargs = parse_arbitrary_arguments(args)
    params = GPTMatcherParams(**kwargs)
    print_dataclass(params)

    cache = init_cache()
    matcher = GPTMatcher(params, verbose=True, cache=cache)
    result = matcher.match(text1, text2)
    print(f"\nСходство: {result}")
