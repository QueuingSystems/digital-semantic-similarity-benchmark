import click
from dss_benchmark.cli.methods import kwm
from dss_benchmark.cli.experiments import kwme

__all__ = ["cli"]


@click.group()
def cli():
    pass


cli.add_command(kwm)
cli.add_command(kwme)

if __name__ == "__main__":
    cli()
