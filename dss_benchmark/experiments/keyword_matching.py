import dataclasses
import time
from multiprocessing.pool import Pool
from multiprocessing.process import current_process
from typing import Dict, List, Union

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
    load_dataset,
    process_auprc,
    process_f1_score,
    process_roc_auc,
)

__all__ = [
    "kwm_match",
    "kwm_match_parallel",
    "kwm_experiment",
    "kwm_measure_timings",
    "kwm_process_timings",
    "kwm_test_mprof",
]


def kwm_match(
    matcher: KeywordDistanceMatcher, cutoff: int, dataset: List[Datum], verbose=False
):
    result: List[ResultDatum] = []
    for datum in tqdm_v(dataset, verbose=verbose):
        value = matcher.match(datum.text_1, datum.text_2)
        result.append(ResultDatum(datum=datum, value=value, match=value >= cutoff))
    return result


def _kwm_match_parallel_init(params, verbose_, cutoff_, cache_kind=None):
    global matcher, cutoff, verbose
    verbose = verbose_
    cache = init_cache(cache_kind)
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
    params: KwDistanceMatcherParams,
    cutoff,
    dataset: List[Datum],
    verbose=False,
    cache_kind: Union[str, None] = None,
):
    result: List[ResultDatum] = []

    with Pool(
        initializer=_kwm_match_parallel_init,
        initargs=(params, verbose, cutoff, cache_kind),
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

<<<<<<< HEAD
=======
    plt.rcParams['savefig.dpi'] = 300
>>>>>>> 40832e5fe0690735883dfea59a5fda3f6e178392
    if dataset_name == "studentor_partner":
        _kwm_auprc_curve(dataset, params, results_folder, verbose)
    else:
        _kwm_roc_curve(dataset, params, results_folder, verbose)


def _kwm_get_big_dataset(verbose=False, one_student=False):
    dataset_studentor = load_dataset("studentor_partner")
    unique_resumes = list(set([datum.text_2 for datum in dataset_studentor]))

    if one_student:
        unique_resumes = unique_resumes[:1]

    dataset_examples = load_dataset("all_examples")
    unique_vacancies = list(
        set(
            [
                *[datum.text_2 for datum in dataset_examples],
                *[datum.text_1 for datum in dataset_studentor],
            ]
        )
    )
    if verbose:
        print("Unique resumes:", len(unique_resumes))
        print("Unique vacancies:", len(unique_vacancies))

    big_dataset: List[Datum] = []
    for i, resume in enumerate(unique_resumes):
        for j, vacancy in enumerate(unique_vacancies):
            big_dataset.append(
                Datum(
                    text_1=resume,
                    text_2=vacancy,
                    title_1=f"Resume {i}",
                    title_2=f"Vacancy {j}",
                    need_match=True,
                )
            )
    if verbose:
        print("Big dataset size:", len(big_dataset))
    return big_dataset


def _kwm_measure_time_on_dataset(
    matcher: KeywordDistanceMatcher, big_dataset: List[Datum], verbose=False
):
    times_data = []
    for datum in tqdm_v(big_dataset, verbose=verbose):
        start, process_start = time.time(), time.process_time()
        matcher.match(datum.text_1, datum.text_2)
        end, process_end = time.time(), time.process_time()
        times_data.append(
            {
                "time": end - start,
                "process_time": process_end - process_start,
                "text_1_len": len(datum.text_1),
                "text_2_len": len(datum.text_2),
            }
        )
    df = pd.DataFrame(times_data)
    return df


<<<<<<< HEAD
def kwm_process_timings():
    df_no_cache = pd.read_csv("_output/kwm_times_no_cache.csv")
    df_cached = pd.read_csv("_output/kwm_times_cached.csv")
    df_changed_vacancy = pd.read_csv("_output/kwm_times_no_cache_changed_vacancy.csv")
    df_changed_resume = pd.read_csv("_output/kwm_times_no_cache_changed_resume.csv")
    df_swapped = pd.read_csv("_output/kwm_times_no_cache_swapped.csv")
=======
def kwm_process_timings(prefix='kwm'):
    df_no_cache = pd.read_csv(f"_output/{prefix}_times_no_cache.csv")
    df_cached = pd.read_csv(f"_output/{prefix}_times_cached.csv")
    df_changed_vacancy = pd.read_csv(f"_output/{prefix}_times_no_cache_changed_vacancy.csv")
    df_changed_resume = pd.read_csv(f"_output/{prefix}_times_no_cache_changed_resume.csv")
    df_swapped = pd.read_csv(f"_output/{prefix}_times_no_cache_swapped.csv")
>>>>>>> 40832e5fe0690735883dfea59a5fda3f6e178392

    avg_no_cache_process_time = df_no_cache["process_time"].mean()
    avg_cached_process_time = df_cached["process_time"].mean()

    print("Avg process time without cache: ", round(avg_no_cache_process_time, 5))
    print("Avg process time with cache: ", round(avg_cached_process_time, 5))

    avg_changed_vacancy_process_time = df_changed_vacancy["process_time"].mean()
    avg_changed_resume_process_time = df_changed_resume["process_time"].mean()

    print(
        "Avg process time without cache (changed vacancy): ",
        round(avg_changed_vacancy_process_time, 5),
    )
    print(
        "Avg process time without cache (changed resume): ",
        round(avg_changed_resume_process_time, 5),
    )

    avg_swapped_process_time = df_swapped["process_time"].mean()

    print(
        "Avg process time without cache (swapped): ",
        round(avg_swapped_process_time, 5),
    )
    plt.rcParams['savefig.dpi'] = 300
    fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 4))
    df_no_cache[["process_time", "text_2_len"]].plot(
        ax=ax1,
        x="text_2_len",
        y="process_time",
        label="No cache",
        kind="scatter",
        logx=True,
        title="Process time vs vacancy length",
    )
    # fig.savefig("_output/kwm_times_no_cache_text_2.png")

    # fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    df_swapped[["process_time", "text_1_len"]].plot(
        ax=ax2,
        x="text_1_len",
        y="process_time",
        label="No cache",
        kind="scatter",
        logx=True,
        title="Process time vs resume length",
    )
    plt.tight_layout()
    for ax in [ax1, ax2]:
        ax.get_legend().set_visible(False)
        ax.grid(0.25)
        ax.set_axisbelow(True)
<<<<<<< HEAD
    fig.savefig("_output/kwm_times_no_cache.png")
=======
    fig.savefig(f"_output/{prefix}_times_no_cache.png")
>>>>>>> 40832e5fe0690735883dfea59a5fda3f6e178392



def kwm_measure_timings(verbose=False):
    big_dataset = _kwm_get_big_dataset(verbose=verbose, one_student=True)

    default_params = KwDistanceMatcherParams(
        is_window=True, kw_saturation=True, swap_texts=True
    )
    cache = init_cache("memory")
    matcher = KeywordDistanceMatcher(default_params, False, cache)

    df_no_cache = _kwm_measure_time_on_dataset(matcher, big_dataset, verbose)
    df_no_cache.to_csv("_output/kwm_times_no_cache.csv", index=False)

    df_cached = _kwm_measure_time_on_dataset(matcher, big_dataset, verbose)
    df_cached.to_csv("_output/kwm_times_cached.csv", index=False)

    big_dataset[
        0
    ].text_2 += ". Этот текст был изменен, чтобы проверить скорость пересчёта"
    df_changed_vacancy = _kwm_measure_time_on_dataset(matcher, big_dataset, verbose)
    df_changed_vacancy.to_csv(
        "_output/kwm_times_no_cache_changed_vacancy.csv", index=False
    )

    for datum in big_dataset:
        datum.text_1 += ". Этот текст был изменен, чтобы проверить скорость пересчёта"
    df_changed_resume = _kwm_measure_time_on_dataset(matcher, big_dataset, verbose)
    df_changed_resume.to_csv(
        "_output/kwm_times_no_cache_changed_resume.csv", index=False
    )

    big_dataset_swapped = []
    for datum in big_dataset:
        big_dataset_swapped.append(
            Datum(text_1=datum.text_2, text_2=datum.text_1, title_1=datum.title_2, title_2=datum.title_1, need_match=datum.need_match)
        )
    df_swapped = _kwm_measure_time_on_dataset(matcher, big_dataset_swapped, verbose)
    df_swapped.to_csv("_output/kwm_times_no_cache_swapped.csv", index=False)


def kwm_test_mprof(parallel=False, verbose=False):
    big_dataset = _kwm_get_big_dataset(verbose=verbose, one_student=True)
    default_params = KwDistanceMatcherParams(
        is_window=True, kw_saturation=True, swap_texts=True
    )
    if parallel:
        kwm_match_parallel(default_params, 0, big_dataset, verbose, 'memory')
    else:
        matcher = KeywordDistanceMatcher(default_params, False, init_cache("memory"))
        kwm_match(matcher, 0, big_dataset, verbose)
