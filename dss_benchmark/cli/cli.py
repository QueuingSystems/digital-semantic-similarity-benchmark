import click
from dss_benchmark.cli.experiments import benchmarks, gptme, kwme, tfidfe
from dss_benchmark.cli.methods import kwm, gpt, tfidf


__all__ = ["cli"]


@click.group()
def cli():
    pass


cli.add_command(kwm)
cli.add_command(kwme)
cli.add_command(benchmarks)
cli.add_command(gptme)
cli.add_command(gpt)
cli.add_command(tfidf)
cli.add_command(tfidfe)

if __name__ == "__main__":
    cli()
