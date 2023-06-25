import click
from dss_benchmark.common import init_cache, parse_arbitrary_arguments, print_dataclass
from dss_benchmark.experiments import (
    DATASETS,
    gpt_match,
    gpt_measure_timings,
    kwm_process_timings,
    load_dataset,
)
from dss_benchmark.methods.chatgpt import GPTMatcher, GPTMatcherParams

from .common import print_results

__all__ = ["gptme"]


@click.group("gpt-matching-exp", help="Эксперименты: сопоставление через промпты к GPT")
def gptme():
    pass


@gptme.command(
    help="Проверить работу на датасете с данными параметрами",
    context_settings=dict(ignore_unknown_options=True),
)
@click.option(
    "-d", "--dataset-name", type=click.Choice(DATASETS), required=True, prompt=True
)
@click.option(
    "-c",
    "--cutoff",
    type=click.IntRange(0, 100, True, True),
    required=True,
    default=50,
)
@click.option("-s", "--shot", type=click.INT, multiple=True, help="Индекс образца")
@click.option(
    "-sv", "--shot-value", type=click.INT, multiple=True, help="Значение образца"
)
@click.argument("args", nargs=-1, type=click.UNPROCESSED)
def match(dataset_name, cutoff, shot, shot_value, args):
    cache = init_cache()
    dataset = load_dataset(dataset_name)
    # random.Random(42).shuffle(dataset)
    kwargs = parse_arbitrary_arguments(args)
    params = GPTMatcherParams(**kwargs)
    print_dataclass(params)
    matcher = GPTMatcher(params, cache=cache, verbose=False)

    assert len(shot) == len(shot_value)

    for idx, value in zip(shot, shot_value):
        datum = dataset[idx]
        matcher.add_shot(datum.text_1, datum.text_2, int(value / 10), tokens=2000)

    results = gpt_match(matcher, cutoff, dataset, verbose=True)
    print_results(results, dataset)


@gptme.command(
    help="Провести эксперимент с временем",
)
def timings():
    gpt_measure_timings(True)


@gptme.command(
    help="Обработать результаты эксперимента со временем",
)
def timings_process():
    kwm_process_timings("gpt")
