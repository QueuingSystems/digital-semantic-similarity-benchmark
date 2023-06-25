import click

from dss_benchmark.experiments import do_grand_total


__all__ = ['gt']

@click.group('gt-exp', help="Эксперименты над всем, что есть")
def gt():
    pass


@gt.command(
    help="Запуск"
)
def run():
    do_grand_total()
