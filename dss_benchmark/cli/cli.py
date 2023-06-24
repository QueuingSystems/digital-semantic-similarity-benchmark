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
<<<<<<< HEAD
cli.add_command(trn)
cli.add_command(mch)
cli.add_command(rsch)

cli.add_command(benchmarks)
cli.add_command(benchmarks)
=======
cli.add_command(benchmarks)
cli.add_command(gptme)
cli.add_command(gpt)
>>>>>>> 40832e5fe0690735883dfea59a5fda3f6e178392

if __name__ == "__main__":
    cli()
