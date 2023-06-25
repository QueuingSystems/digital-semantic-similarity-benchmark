import click
from dss_benchmark.experiments.grand_total import (
    do_grand_total,
    do_timings_total,
    process_timings,
    run_for_mprof
)

__all__ = ["gt"]


@click.group("gt-exp", help="Эксперименты над всем, что есть")
def gt():
    pass


@gt.command(help="Запуск")
def run():
    do_grand_total()


@gt.command(help="Изменение времени")
def timings():
    do_timings_total()


@gt.command(help="Обработка результатов измерения времени")
def timings_process():
    process_timings()


@gt.command(help='Запуск для mprof')
@click.option(
    "-m",
    "--model-name",
    required=True,
    type=str,
    help="Название модели",
    prompt=True,
)
@click.option(
    '-nc',
    '--no-cache',
    is_flag=True,
    help='Не использовать кэш',
)
def run_mprof(model_name, no_cache):
    run_for_mprof(model_name, no_cache)
