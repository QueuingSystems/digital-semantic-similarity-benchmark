import csv
import json
from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn import metrics

__all__ = [
    "Datum",
    "load_dataset",
    "DATASETS",
    "ResultDatum",
    "confusion_matrix",
    "f1_score",
    "process_roc_auc",
    "process_f1_score",
    "process_auprc",
]


DATASETS = ["rpd_dataset", "all_examples", "studentor_partner"]

DATASETS = ["rpd_dataset", "all_examples", "studentor_partner", "dataset_v6_r30"]


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
    if name == "rpd_dataset" or name == "all_examples":
        with open(f"./data/{name}.json", "r") as f:
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
    if name == "studentor_partner":
        with open(f"./data/{name}.csv", "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return [
            Datum(
                text_1=datum["text_1"],
                text_2=datum["text_2"],
                title_1=datum["title_1"],
                title_2=datum["title_2"],
                need_match=datum["need_match"] == "TRUE",
            )
            for datum in data
        ]

    if name == "dataset_v6_r30":
        with open(f"./data/{name}.csv", "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return [
            Datum(
                text_1=datum["vacancy"],
                text_2=datum["resume"],
                title_1=f"vacancy_{i}",
                title_2=f"resume_{i}",
                need_match=datum["result"] == "1",
            )
            for i, datum in enumerate(data)
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


def process_roc_auc(dataset: List[Datum], results: List[ResultDatum]):
    y_true = [int(datum.need_match) for datum in dataset]
    y_score = [datum.value for datum in results]

    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    cutoff = thresholds[np.argmax(tpr - fpr)]
    auc = metrics.auc(fpr, tpr)

    return fpr, tpr, cutoff, auc


def process_auprc(dataset: List[Datum], results: List[ResultDatum]):
    y_true = [int(datum.need_match) for datum in dataset]
    y_score = [datum.value for datum in results]

    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
    auprc = metrics.auc(recall, precision)

    f_score = 2 * precision * recall / (precision + recall)
    best_f_score_idx = np.argmax(f_score)
    best_f_score = f_score[best_f_score_idx]
    best_cutoff = thresholds[best_f_score_idx]

    return precision, recall, auprc, best_cutoff, best_f_score


def process_f1_score(results: List[ResultDatum]):
    cutoff_values = np.linspace(0, 100, 1000)
    f1_values = []
    for i in cutoff_values:
        for result in results:
            result.match = result.value >= i
        tp, fp, fn, tn = confusion_matrix(results)
        f1 = f1_score(tp, fp, fn)
        f1_values.append(f1)
    best_f1_idx = np.argmax(f1_values)
    best_f1 = f1_values[best_f1_idx]
    best_cutoff = cutoff_values[best_f1_idx]
    return best_f1, best_cutoff
