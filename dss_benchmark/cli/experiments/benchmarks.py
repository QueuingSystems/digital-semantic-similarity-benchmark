import numpy as np
import click

from dss_benchmark.experiments import DATASETS, load_dataset

__all__ = ["benchmarks"]


@click.group(name="benchmarks", help="Данные бенчмарков")
def benchmarks():
    pass


@benchmarks.command(help="Посмотреть параметры бенчмарка")
@click.option(
    "-d", "--dataset-name", type=click.Choice(DATASETS), required=True, prompt=True
)
def params(dataset_name):
    dataset = load_dataset(dataset_name)

    should_match = len([x for x in dataset if x.need_match])
    print(f"Length: {len(dataset)}")
    print(f"Should match: {should_match}")

    unique_texts_1 = set([x.text_1 for x in dataset])
    unique_texts_2 = set([x.text_2 for x in dataset])
    print(f"Unique texts 1: {len(unique_texts_1)}")
    print(f"Unique texts 2: {len(unique_texts_2)}")

    mean_text_length_1 = np.mean([len(x.text_1) for x in dataset])
    mean_text_length_2 = np.mean([len(x.text_2) for x in dataset])
    iqr_text_length_1 = np.percentile(
        [len(x.text_1) for x in dataset], 75
    ) - np.percentile([len(x.text_1) for x in dataset], 25)
    iqr_text_length_2 = np.percentile(
        [len(x.text_2) for x in dataset], 75
    ) - np.percentile([len(x.text_2) for x in dataset], 25)
    print(f"Mean text length 1: {mean_text_length_1}")
    print(f"Mean text length 2: {mean_text_length_2}")
    print(f"IQR text length 1: {iqr_text_length_1}")
    print(f"IQR text length 2: {iqr_text_length_2}")
