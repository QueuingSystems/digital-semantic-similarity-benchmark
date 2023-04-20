import json
from dataclasses import dataclass
from typing import List

__all__ = [
    "Datum",
    "load_dataset",
    "DATASETS",
    "ResultDatum",
    "confusion_matrix",
    "f1_score",
]


DATASETS = ["rpd_dataset"]


@dataclass
class Datum:
    text_1: str
    text_2: str
    title_1: str
    title_2: str
    need_match: bool


@dataclass
class ResultDatum:
    datum: Datum
    value: float
    match: bool


def load_dataset(name: str) -> List[Datum]:
    if name == "rpd_dataset":
        with open("./data/rpd_dataset.json", "r") as f:
            data = json.load(f)
        return [
            Datum(
                text_1=datum["text_rp"],
                text_2=datum["text_proj"],
                title_1=f'{datum["id_rp"]}: {datum["name_rp"]}',
                title_2=f'{datum["id_proj"]}: {datum["name_proj"]}',
                need_match=datum["need_match"],
            )
            for datum in data
        ]
    raise ValueError(f"Unknown dataset: {name}")


def confusion_matrix(results: List[ResultDatum]):
    tp, fp, fn, tn = 0, 0, 0, 0
    for result in results:
        if result.datum.need_match and result.match:
            tp += 1
        elif result.datum.need_match and not result.match:
            fp += 1
        elif not result.datum.need_match and result.match:
            fn += 1
        elif not result.datum.need_match and not result.match:
            tn += 1
    return (tp, fp, fn, tn)


def f1_score(tp: int, fp: int, fn: int):
    return (2 * tp) / (2 * tp + fp + fn)
