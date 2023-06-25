import dataclasses
from typing import List

import numpy as np
import pandas as pd
from dss_benchmark.common import tqdm_v
from dss_benchmark.common.dataclass_utils import print_dataclass
from dss_benchmark.methods.tfidf import TfIdfMatcher, TfIdfMatcherParams
from matplotlib import pyplot as plt

from .common import (
    Datum,
    ResultDatum,
    confusion_matrix,
    f1_score,
    process_auprc,
    process_f1_score,
    process_roc_auc,
)

__all__ = [
    "tfidf_match",
    "tfidf_experiment",
]


def tfidf_match(matcher: TfIdfMatcher, cutoff: int, dataset: List[Datum], verbose=False):
    result: List[ResultDatum] = []
    for datum in tqdm_v(dataset, verbose=verbose):
        value = matcher.match(datum.text_1, datum.text_2)
        result.append(ResultDatum(datum=datum, value=value, match=value >= cutoff))
    return result


def tfidf_experiment(
    dataset: List[Datum], dataset_name: str, results_folder: str, verbose=False
):
    key = "auc"
    if dataset_name == "studentor_partner":
        key = "auprc"
    result1 = _research_ngram(dataset, dataset_name, key, results_folder, verbose)
    params1 = TfIdfMatcherParams(
        train_data="all_examples",
        min_ngram=result1["ngram"][0],
        max_ngram=result1["ngram"][1],
    )
    result2 = _research_bin_and_sublinear_tf(
        dataset, params1, key, dataset_name, results_folder, verbose
    )
    params2 = params1
    params2.binary = result2["binary"]
    params2.sublinear_tf = result2["sublinear_tf"]
    params = [params1, params2]
    results = [result1, result2]
    best_values = []
    for param, datum in zip(params, results):
        if verbose:
            print_dataclass(param)
            print(datum)
        value = {**dataclasses.asdict(param), **datum.to_dict()}
        best_values.append(value)
    df = pd.DataFrame(best_values)
    df.to_csv("{}/tfidf {}.csv".format(results_folder, dataset_name))
    if dataset_name == "studentor_partner":
        _tfidf_auprc_curve(dataset, params, results_folder, verbose)
    else:
        _tfidf_roc_curve(dataset, params, results_folder, verbose)


def _research_ngram(
    dataset: List[Datum],
    dataset_name: str,
    key: str,
    results_folder: str,
    verbose=False,
):
    data = []
    list_ngram = [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]
    for ngram in list_ngram:
        params = TfIdfMatcherParams(
            train_data="all_examples", min_ngram=ngram[0], max_ngram=ngram[1]
        )
        model = TfIdfMatcher(params)
        for cutoff in np.linspace(0, 100, 100):
            result = tfidf_match(model, cutoff, dataset)
            data.append(
                {**_get_results(dataset, result), "cutoff": cutoff, "ngram": ngram}
            )
    df = pd.DataFrame(data)
    df.to_csv("{}/tfidf_{}.csv".format(results_folder, dataset_name))
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    groups = df[[key, "cutoff", "ngram"]].groupby("ngram")
    for _, group in groups:
        group = group.reset_index()
        group[[key, "cutoff", "ngram"]].plot(
            x="cutoff",
            y=key,
            kind="line",
            ax=ax,
            label="ngram=({},{})".format(group["ngram"][0], group["ngram"][1]),
        )
    ax.set_xlabel("tf-idf cutoff")
    ax.set_ylabel(key)
    ax.set_title("binary=False sublinear_tf=False")
    ax.legend()
    ax.grid(0.25)
    fig.savefig(f"{results_folder}/tf-idf_1_{key}.png")
    ##графики
    best_ngram = df.iloc[df[key].idxmax()]
    return best_ngram


def _research_bin_and_sublinear_tf(
    dataset: List[Datum],
    params: TfIdfMatcherParams,
    key: str,
    dataset_name: str,
    results_folder: str,
    verbose=False,
):
    data = []
    list_bin = [False, True]
    list_sublinear = [False, True]
    for bin in list_bin:
        for sub in list_sublinear:
            params.binary = bin
            params.sublinear_tf = sub
            model = TfIdfMatcher(params)
            for cutoff in np.linspace(0, 100, 100):
                result = tfidf_match(model, cutoff, dataset)
                data.append(
                    {
                        **_get_results(dataset, result),
                        "cutoff": cutoff,
                        "binary": bin,
                        "sublinear_tf": sub,
                    }
                )
    df = pd.DataFrame(data)
    df.to_csv("{}/tfidf_{}_ngram.csv".format(results_folder, dataset_name))
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    groups = df[[key, "cutoff", "binary", "sublinear_tf"]].groupby(
        ["binary", "sublinear_tf"]
    )
    for _, group in groups:
        group = group.reset_index()
        group[[key, "cutoff", "binary", "sublinear_tf"]].plot(
            x="cutoff",
            y=key,
            kind="line",
            ax=ax,
            label="binary={} sublinear_tf={}".format(
                group["binary"][0], group["sublinear_tf"][0]
            ),
        )
    ax.set_xlabel("tf-idf cutoff")
    ax.set_ylabel(key)
    ax.set_title("ngram=({},{})".format(params.min_ngram, params.max_ngram))
    ax.legend()
    ax.grid(0.25)
    fig.savefig(f"{results_folder}/tf-idf_2_{key}.png")
    best_ngram = df.iloc[df[key].idxmax()]
    return best_ngram


def _get_results(dataset: List[Datum], results: List[ResultDatum]):
    _, _, auc_cutoff, auc = process_roc_auc(dataset, results)
    for r in results:
        r.match = r.value >= auc_cutoff
    tp, fp, fn, tn = confusion_matrix(results)
    auc_f1 = f1_score(tp, fp, fn)
    f1, cutoff = process_f1_score(results)

    _, _, auprc, auprc_cutoff, auprc_f1 = process_auprc(dataset, results)

    return {
        "auc": auc,
        "auc_cutoff": auc_cutoff,
        "auc_f1": auc_f1,
        "f1": f1,
        "f1_cutoff": cutoff,
        "auprc": auprc,
        "auprc_cutoff": auprc_cutoff,
        "auprc_f1": auprc_f1,
    }


def _tfidf_roc_curve(
    dataset: List[Datum],
    params: List[TfIdfMatcherParams],
    results_folder: str,
    verbose=False,
):
    options = [
        (
            param,
            "tf-idf ngram=({},{}) binary={} sublinear-tf={}".format(
                param.min_ngram, param.max_ngram, param.binary, param.sublinear_tf
            ),
        )
        for param in params
    ]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for param, title in options:
        model = TfIdfMatcher(param)
        result = tfidf_match(model, 0, dataset, verbose=False)
        fpr, tpr, cutoff, auc = process_roc_auc(dataset, result)
        ax.plot(fpr, tpr, label=f"{title} (AUC={auc:.3f}, cutoff={cutoff:.1f})")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("tf-idf matcher ROC curves")
    ax.legend(loc="lower right")
    ax.grid(0.25)
    fig.savefig(f"{results_folder}/tfidf_agg_auc.png")


def _tfidf_auprc_curve(
    dataset: List[Datum],
    params: List[TfIdfMatcherParams],
    results_folder: str,
    verbose=False,
):
    options = [
        (
            param,
            "tf-idf ngram=({},{}) binary={} sublinear-tf={}".format(
                param.min_ngram, param.max_ngram, param.binary, param.sublinear_tf
            ),
        )
        for param in params
    ]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for param, title in options:
        model = TfIdfMatcher(param)
        result = tfidf_match(model, 0, dataset, verbose=False)
        precision, recall, auprc, auprc_cutoff, auprc_f1 = process_auprc(
            dataset, result
        )
        ax.plot(recall, precision, label=f"{title} (AUPRC={auprc:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("tf-idf matcher AUPRC curves")
    ax.legend()
    ax.grid(0.25)
    fig.savefig(f"{results_folder}/tfidf_agg_auprc.png")
