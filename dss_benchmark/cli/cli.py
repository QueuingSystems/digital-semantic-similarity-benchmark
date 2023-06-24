import click
from dss_benchmark.cli.experiments import benchmarks, gptme, kwme
from dss_benchmark.cli.methods import kwm, gpt

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
cli.add_command(benchmarks)
cli.add_command(gptme)
cli.add_command(gpt)

cli.add_command(trn)
cli.add_command(mch)
cli.add_command(rsch)

if __name__ == "__main__":
    cli()
