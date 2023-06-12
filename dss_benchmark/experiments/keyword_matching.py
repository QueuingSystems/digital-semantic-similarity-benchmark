import dataclasses
import time
from multiprocessing.pool import Pool
from multiprocessing.process import current_process
from typing import Dict, List

import numpy as np
import pandas as pd
from dss_benchmark.common import init_cache, tqdm_v
from dss_benchmark.common.dataclass_utils import print_dataclass
from dss_benchmark.methods.keyword_matching import (
    KeywordDistanceMatcher,
    KwDistanceMatcherParams,
)
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

__all__ = ["kwm_match", "kwm_match_parallel", "kwm_experiment"]


def kwm_match(
    matcher: KeywordDistanceMatcher, cutoff: int, dataset: List[Datum], verbose=False
):
    result: List[ResultDatum] = []
    for datum in tqdm_v(dataset, verbose=verbose):
        value = matcher.match(datum.text_1, datum.text_2)
        result.append(ResultDatum(datum=datum, value=value, match=value >= cutoff))
    return result


def _kwm_match_parallel_init(params, verbose_, cutoff_):
    global matcher, cutoff, verbose
    verbose = verbose_
    cache = init_cache()
    matcher = KeywordDistanceMatcher(params, False, cache)
    cutoff = cutoff_
    if verbose:
        process = current_process()
        print(f"Initialized worker: {process.name} ({process.pid})")


def _kwm_match_task(datum: Datum):
    global matcher, cutoff, verbose
    start_time = time.time()
    value = matcher.match(datum.text_1, datum.text_2)
    end_time = time.time()
    if verbose:
        process = current_process()
        print(
            f"Finished worker: {process.name} ({process.pid}) in {end_time - start_time:.2f}s"
        )
    return ResultDatum(datum=datum, value=value, match=value >= cutoff)


def kwm_match_parallel(
    params: KwDistanceMatcherParams, cutoff, dataset: List[Datum], verbose=False
):
    result: List[ResultDatum] = []

    with Pool(
        initializer=_kwm_match_parallel_init,
        initargs=(params, verbose, cutoff),
    ) as pool:
        for datum in tqdm_v(
            pool.map(_kwm_match_task, dataset),
            total=len(dataset),
            verbose=verbose,
        ):
            result.append(datum)
    return result


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


def _kwm_try_1(dataset: List[Datum], results_folder: str, key="auc", verbose=False):
    base_1 = KwDistanceMatcherParams(
        is_window=False, kw_saturation=True, swap_texts=False
    )
    data_1 = []
    for dbscan_eps in tqdm_v(np.arange(0.6, 1.1, 0.1), verbose=verbose):
        base_1.dbscan_eps = dbscan_eps
        for kw_cutoff in np.linspace(0, 1, 20):
            base_1.kw_cutoff = kw_cutoff
            result = kwm_match_parallel(base_1, 0, dataset, verbose=False)
            data_1.append(
                {
                    **_get_results(dataset, result),
                    "kw_cutoff": kw_cutoff,
                    "dbscan_eps": dbscan_eps,
                }
            )
    df_1 = pd.DataFrame(data_1)
    df_1.to_csv(f"{results_folder}/kwm_1.csv", index=False)
    # df_1 = pd.read_csv(f"{results_folder}/kwm_1.csv")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    groups = df_1[[key, "kw_cutoff", "dbscan_eps"]].groupby("dbscan_eps")
    for _, group in groups:
        group = group.reset_index()
        group[[key, "kw_cutoff", "dbscan_eps"]].plot(
            x="kw_cutoff",
            y=key,
            kind="line",
            ax=ax,
            label=f"DBSCAN eps={group['dbscan_eps'][0]:.2f}",
        )
    ax.set_xlabel("Keyword cutoff")
    ax.set_ylabel(key)
    ax.set_title("is_window=False, kw_saturation=True, swap_texts=False")
    ax.legend()
    ax.grid(0.25)
    fig.savefig(f"{results_folder}/kwm_1_{key}.png")

    best_datum = df_1.iloc[df_1[key].idxmax()]
    base_1.dbscan_eps = best_datum["dbscan_eps"]
    base_1.kw_cutoff = best_datum["kw_cutoff"]
    return base_1, best_datum


def _kwm_try_2(dataset: List[Datum], results_folder: str, key="auc", verbose=False):
    base_2 = KwDistanceMatcherParams(
        is_window=True, kw_saturation=True, swap_texts=False
    )
    data_2 = []
    for window_size in tqdm_v([170, 400, 500, 700], verbose=verbose):
        base_2.window_size = window_size
        for kw_cutoff in np.linspace(0, 1, 20):
            base_2.kw_cutoff = kw_cutoff
            result = kwm_match_parallel(base_2, 0, dataset, verbose=False)
            data_2.append(
                {
                    **_get_results(dataset, result),
                    "kw_cutoff": kw_cutoff,
                    "window_size": window_size,
                }
            )
    df_2 = pd.DataFrame(data_2)
    df_2.to_csv(f"{results_folder}/kwm_2.csv", index=False)
    # df_2 = pd.read_csv(f"{results_folder}/kwm_2.csv")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    groups = df_2[[key, "kw_cutoff", "window_size"]].groupby("window_size")
    for _, group in groups:
        group = group.reset_index()
        group[[key, "kw_cutoff", "window_size"]].plot(
            x="kw_cutoff",
            y=key,
            kind="line",
            ax=ax,
            label=f"Window size={group['window_size'][0]}",
        )
    ax.set_xlabel("Keyword cutoff")
    ax.set_ylabel(key)
    ax.set_title(
        "is_window=True, kw_saturation=True, swap_texts=False, dbscan_eps=0.95"
    )
    ax.legend()
    ax.grid(0.25)
    fig.savefig(f"{results_folder}/kwm_2_{key}.png")

    best_datum = df_2.iloc[df_2[key].idxmax()]
    base_2.window_size = best_datum["window_size"]
    base_2.kw_cutoff = best_datum["kw_cutoff"]
    return base_2, best_datum


def _kwm_try_3(dataset: List[Datum], results_folder: str, key="auc", verbose=False):
    base_3 = KwDistanceMatcherParams(
        is_window=True, kw_saturation=True, swap_texts=True
    )
    data_3 = []
    for window_size in tqdm_v([170, 300, 400, 500, 700], verbose=verbose):
        base_3.window_size = window_size
        for kw_cutoff in np.linspace(0, 1, 20):
            base_3.kw_cutoff = kw_cutoff
            result = kwm_match_parallel(base_3, 0, dataset, verbose=False)
            data_3.append(
                {
                    **_get_results(dataset, result),
                    "kw_cutoff": kw_cutoff,
                    "window_size": window_size,
                }
            )
    df_3 = pd.DataFrame(data_3)
    df_3.to_csv(f"{results_folder}/kwm_3.csv", index=False)
    # df_3 = pd.read_csv(f"{results_folder}/kwm_3.csv")

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    groups = df_3[[key, "kw_cutoff", "window_size"]].groupby("window_size")
    for _, group in groups:
        group = group.reset_index()
        group[[key, "kw_cutoff", "window_size"]].plot(
            x="kw_cutoff",
            y=key,
            kind="line",
            ax=ax,
            label=f"Window size={group['window_size'][0]}",
        )
    ax.set_xlabel("Keyword cutoff")
    ax.set_ylabel(key)
    ax.set_title("is_window=True, kw_saturation=True, swap_texts=True, dbscan_eps=0.95")
    ax.legend()
    ax.grid(0.25)
    fig.savefig(f"{results_folder}/kwm_3_{key}.png")

    best_datum = df_3.iloc[df_3[key].idxmax()]
    base_3.window_size = best_datum["window_size"]
    base_3.kw_cutoff = best_datum["kw_cutoff"]
    return base_3, best_datum


def _kwm_roc_curve(
    dataset: List[Datum],
    params: List[KwDistanceMatcherParams],
    results_folder: str,
    verbose=False,
):
    options = [
        (param, f"is_window={param.is_window}, swap_texts={param.swap_texts}")
        for param in params
    ]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for param, title in options:
        result = kwm_match_parallel(param, 0, dataset, verbose=False)
        fpr, tpr, cutoff, auc = process_roc_auc(dataset, result)
        ax.plot(fpr, tpr, label=f"{title} (AUC={auc:.3f}, cutoff={cutoff:.1f})")

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Keyword matcher ROC curves")
    ax.legend(loc="lower right")
    ax.grid(0.25)
    fig.savefig(f"{results_folder}/kwm_agg_auc.png")


def _kwm_auprc_curve(
    dataset: List[Datum],
    params: List[KwDistanceMatcherParams],
    results_folder: str,
    verbose=False,
):
    options = [
        (param, f"is_window={param.is_window}, swap_texts={param.swap_texts}")
        for param in params
    ]
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    for param, title in options:
        results = kwm_match_parallel(param, 0, dataset, verbose=False)
        precision, recall, auprc, auprc_cutoff, auprc_f1 = process_auprc(
            dataset, results
        )
        ax.plot(recall, precision, label=f"{title} (AUPRC={auprc:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Keyword matcher AUPRC curves")
    ax.legend()
    ax.grid(0.25)
    fig.savefig(f"{results_folder}/kwm_agg_auprc.png")


def kwm_experiment(
    dataset: List[Datum], dataset_name: str, results_folder: str, verbose=False
):
    key = "auc"
    if dataset_name == "studentor_partner":
        key = "auprc"
    best_1, datum_1 = _kwm_try_1(dataset, results_folder, key, verbose)
    best_2, datum_2 = _kwm_try_2(dataset, results_folder, key, verbose)
    best_3, datum_3 = _kwm_try_3(dataset, results_folder, key, verbose)
    params = [best_1, best_2, best_3]
    data = [datum_1, datum_2, datum_3]
    best_values = []
    for param, datum in zip(params, data):
        if verbose:
            print_dataclass(param)
            print(datum)
        value = {**dataclasses.asdict(param), **datum.to_dict()}
        best_values.append(value)

    df = pd.DataFrame(best_values)
    df.to_csv(f"{results_folder}/kwm_best.csv", index=False)

    if dataset_name == "studentor_partner":
        _kwm_auprc_curve(dataset, params, results_folder, verbose)
    else:
        _kwm_roc_curve(dataset, params, results_folder, verbose)
