import click
from dss_benchmark.cli.methods import kwm

__all__ = ["cli"]


@click.group()
def cli():
    pass


cli.add_command(kwm)

if __name__ == "__main__":
    cli()
