import click
from dss_benchmark.cli.methods import kwm
from dss_benchmark.cli.experiments import kwme

from dss_benchmark.cli.methods import trn
from dss_benchmark.cli.methods import mch
from dss_benchmark.cli.methods import rsch
from dss_benchmark.cli.experiments import benchmarks

from dss_benchmark.cli.experiments import benchmarks

__all__ = ["cli"]


@click.group()
def cli():
    pass


cli.add_command(kwm)
cli.add_command(kwme)
cli.add_command(trn)
cli.add_command(mch)
cli.add_command(rsch)

cli.add_command(benchmarks)

cli.add_command(benchmarks)

if __name__ == "__main__":
    cli()
