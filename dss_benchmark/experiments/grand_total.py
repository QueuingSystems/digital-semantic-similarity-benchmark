import hashlib
import os
import time
from typing import Any, Dict, List

import pandas as pd
from dss_benchmark.common import init_cache, tqdm_v
from dss_benchmark.experiments.common import (
    Datum,
    ResultDatum,
    confusion_matrix,
    f1_score,
    load_dataset,
    process_auprc,
    process_f1_score,
    process_roc_auc,
)
from dss_benchmark.experiments.keyword_matching import kwm_match_parallel
from dss_benchmark.methods import AbstractSimilarityMethod
from dss_benchmark.methods.anmatveev import (
    ANMMatchManager,
    ANMTrainModelManager,
    ANMTrainModelParams,
)
from dss_benchmark.methods.chatgpt import GPTMatcher, GPTMatcherParams
from dss_benchmark.methods.keyword_matching import (
    KeywordDistanceMatcher,
    KwDistanceMatcherParams,
)
from dss_benchmark.methods.tfidf.tfidf import TfIdfMatcher, TfIdfMatcherParams
from matplotlib import pyplot as plt

from .keyword_matching import _kwm_get_big_dataset, _kwm_measure_time_on_dataset

__all__ = ["do_grand_total", "do_timings_total", "process_timings", "run_for_mprof"]

MODELS = {
    "keywords": KeywordDistanceMatcher,
    "gpt": GPTMatcher,
    "tfidf": TfIdfMatcher,
    "word2vec": ANMMatchManager,
    "fastText": ANMMatchManager,
    "paraphrase-multilingual-MiniLM-L12-v2": ANMMatchManager,
    "DeepPavlov/rubert-base-cased": ANMMatchManager,
}

GPT_SHOTS = {
    "all_examples": [(13, 100), (28, 100)],
    "studentor_partner": [(18, 90), (110, 100)],
    "dataset_v6_r30": [],
}

TRAINING_TEXTS = "data/documents_preprocessed.json"
BEST_TRAINING_PARAMS = {
    "all_examples": {
        "fastText": ANMTrainModelParams(
            sg=1, window=15, epochs=6, min_count=5, vector_size=50, min_n=3, max_n=6
        ),
        "word2vec": ANMTrainModelParams(
            sg=1, window=30, epochs=8, min_count=5, vector_size=50
        ),
    },
    "studentor_partner": {
        "fastText": ANMTrainModelParams(
            sg=1, window=10, epochs=5, min_count=5, vector_size=50, min_n=3, max_n=6
        ),
        "word2vec": ANMTrainModelParams(
            sg=1, window=15, epochs=5, min_count=6, vector_size=50
        ),
    },
    "dataset_v6_r30": {
        "fastText": ANMTrainModelParams(
            sg=1, window=15, epochs=5, min_count=5, vector_size=50, min_n=3, max_n=6
        ),
        "word2vec": ANMTrainModelParams(
            sg=1, window=30, epochs=8, min_count=5, vector_size=50
        ),
    },
}

BEST_PARAMS = {
    "all_examples": {
        "keywords": KwDistanceMatcherParams(
            is_window=True,
            window_size=400,
            window_delta=50,
            kw_saturation=True,
            saturate_value=6,
            dbscan_eps=0.95,
            kw_cutoff=0,
            swap_texts=True,
        ),
        "gpt": GPTMatcherParams(
            max_tokens=10000, temperature=0.75, chat_model="gpt-3.5-turbo-16k"
        ),
        "tfidf": TfIdfMatcherParams(
            train_data="all_examples",
            min_ngram=1,
            max_ngram=1,
            binary=False,
            sublinear_tf=False,
        ),
    },
    "studentor_partner": {
        "keywords": KwDistanceMatcherParams(
            is_window=True,
            window_size=170,
            window_delta=50,
            kw_saturation=True,
            saturate_value=6,
            dbscan_eps=0.95,
            kw_cutoff=0.89,
            swap_texts=True,
        ),
        "gpt": GPTMatcherParams(
            max_tokens=10000, temperature=0.75, chat_model="gpt-3.5-turbo-16k"
        ),
        "tfidf": TfIdfMatcherParams(
            train_data="all_examples",
            min_ngram=3,
            max_ngram=3,
            binary=False,
            sublinear_tf=False,
        ),
    },
    "dataset_v6_r30": {
        "keywords": KwDistanceMatcherParams(
            is_window=True,
            window_size=400,
            window_delta=50,
            kw_saturation=True,
            saturate_value=6,
            dbscan_eps=0.95,
            kw_cutoff=0.57,
            swap_texts=False,
        ),
        "gpt": GPTMatcherParams(
            max_tokens=10000, temperature=0.75, chat_model="gpt-3.5-turbo-16k"
        ),
        "tfidf": TfIdfMatcherParams(
            train_data="all_examples",
            min_ngram=1,
            max_ngram=1,
            binary=False,
            sublinear_tf=False,
        ),
    },
}


def _prepare_anns():
    for dataset in BEST_TRAINING_PARAMS.keys():
        for model in ["word2vec", "fastText"]:
            params = BEST_TRAINING_PARAMS[dataset][model]
            params.texts = TRAINING_TEXTS
            params_hash = hashlib.md5(str(params).encode()).hexdigest()
            model_path = f"_output/models/{model}-{params_hash}"
            if not os.path.exists(model_path):
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                print(f"Training {model} for {dataset}")
                trainer = ANMTrainModelManager(params)
                trainer.train(model, model_path)


def _get_model(dataset_name: str, model_name: str, cache, verbose=False):
    params = BEST_PARAMS[dataset_name].get(model_name, None)
    dataset = load_dataset(dataset_name)
    if model_name in ["word2vec", "fastText"]:
        train_params = BEST_TRAINING_PARAMS[dataset_name][model_name]
        train_params.texts = TRAINING_TEXTS
        train_params_hash = hashlib.md5(str(train_params).encode()).hexdigest()
        model_path = f"_output/models/{model_name}-{train_params_hash}"
        matcher = ANMMatchManager(model_path, train_params, cache=cache)
        matcher.set_current_model(model_path)
    elif model_name in [
        "paraphrase-multilingual-MiniLM-L12-v2",
        "DeepPavlov/rubert-base-cased",
    ]:
        matcher = ANMMatchManager(
            model_path=model_name, cache=cache, model_type="transformer"
        )
        matcher.set_current_model(model_name)
        if dataset_name != "all_examples":
            for datum in dataset:
                datum.text_1, datum.text_2 = datum.text_2, datum.text_1
    else:
        Model = MODELS[model_name]
        matcher = Model(params, cache=cache)
    return matcher, dataset


def _do_match(dataset_name: str, model_name: str, cache, verbose=False, parallel=False):
    if model_name == "keywords" and parallel:
        return (
            kwm_match_parallel(
                BEST_PARAMS[dataset_name].get(model_name, None),
                0,
                load_dataset(dataset_name),
                verbose=verbose,
            ),
            None,
        )

    matcher, dataset = _get_model(dataset_name, model_name, cache, verbose)
    result: List[ResultDatum] = []
    times_data = []

    if model_name == "gpt":
        shots = GPT_SHOTS.get(dataset_name, [])
        for idx, val in shots:
            matcher.add_shot(
                dataset[idx].text_1, dataset[idx].text_2, int(val / 10), tokens=2000
            )

    for datum in tqdm_v(dataset, verbose=verbose):
        start, process_start = time.time(), time.process_time()
        value = matcher.match(datum.text_1, datum.text_2)
        end, process_end = time.time(), time.process_time()
        result.append(ResultDatum(datum=datum, value=value, match=value >= 0))
        times_data.append(
            {
                "time": end - start,
                "process_time": process_end - process_start,
                "text_1_len": len(datum.text_1),
                "text_2_len": len(datum.text_2),
            }
        )
    times_df = pd.DataFrame(times_data)
    return result, times_df


def _process_results(results: List[ResultDatum], dataset_name: str):
    dataset = load_dataset(dataset_name)

    f1, cutoff = process_f1_score(results)
    print(f"Optimal cutoff: {cutoff:.4f}, F1: {f1:.4f}")

    for result in results:
        result.match = result.value >= cutoff

    df = pd.DataFrame(
        [
            {
                "title_1": result.datum.title_1,
                "title_2": result.datum.title_2,
                "value": result.value,
                "need_match": result.datum.need_match,
                "match": result.match,
            }
            for result in results
        ]
    )
    print(df)

    tp, fp, fn, tn = confusion_matrix(results)
    f1 = f1_score(tp, fp, fn)

    print(f"TP: {str(tp):3s} FP: {str(fp):3s}")
    print(f"FN: {str(fn):3s} TN: {str(tn):3s}")
    print(f"F1: {f1:.4f}")

    precision, recall, auprc, auprc_cutoff, auprc_f1 = process_auprc(dataset, results)
    print(f"AUPRC: {auprc:.4f}, cutoff: {auprc_cutoff:.4f}, F1:{auprc_f1:.4f}")

    fpr, tpr, auc_cutoff, auc = process_roc_auc(dataset, results)
    print(f"AUC: {auc:.4f}, cutoff: {auc_cutoff:.4f}")
    return {
        "f1": f1,
        "cutoff": cutoff,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "auprc": auprc,
        "auprc_cutoff": auprc_cutoff,
        "auprc_f1": auprc_f1,
        "auc": auc,
        "auc_cutoff": auc_cutoff,
        "precision": precision,
        "recall": recall,
        "fpr": fpr,
        "tpr": tpr,
    }


def _plot_grand_total(results: List[Dict[str, Any]]):
    plt.rcParams["savefig.dpi"] = 300
    results_by_dataset = {}

    for result in results:
        dataset = result["dataset"]
        try:
            results_by_dataset[dataset].append(result)
        except KeyError:
            results_by_dataset[dataset] = [result]

    for dataset, res in results_by_dataset.items():
        if dataset == "studentor_partner" and False:
            res = sorted(res, key=lambda r: r["auprc"], reverse=True)
        else:
            res = sorted(res, key=lambda r: r["auc"], reverse=True)

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        for result in res:
            if dataset == "studentor_partner" and False:
                ax.plot(
                    result["recall"],
                    result["precision"],
                    label=f"{result['model']} (AUPRC={result['auprc']:.4f})",
                )
            else:
                ax.plot(
                    result["fpr"],
                    result["tpr"],
                    label=f"{result['model']} (AUC={result['auc']:.4f})",
                )
            if dataset == "studentor_partner" and False:
                ax.set_title(f"Precision-Recall Curve ({dataset})")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
            else:
                ax.set_title(f"ROC Curve ({dataset})")
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
            ax.legend()
            ax.grid(0.25)
        fig.tight_layout()
        fig.savefig(f"_output/{dataset}_grand_total.png")


def do_grand_total():
    results = []
    cache = init_cache()
    _prepare_anns()
    dataset_names = list(BEST_PARAMS.keys())
    for dataset_name in dataset_names:
        print(f"Dataset: {dataset_name}")
        for model_name in MODELS.keys():
            print(f"Model: {model_name}")
            match_result, _ = _do_match(
                dataset_name, model_name, cache, verbose=True, parallel=True
            )
            result = _process_results(match_result, dataset_name)
            results.append(
                {
                    "dataset": dataset_name,
                    "model": model_name,
                    **result,
                }
            )
    df = pd.DataFrame(results)
    df = df.drop(columns=["precision", "recall", "fpr", "tpr"])
    df.to_csv("_output/grand_total.csv", index=False)

    _plot_grand_total(results)


def _measure_timings(matcher: AbstractSimilarityMethod, model_name: str, verbose=False):
    os.makedirs("_output/timings", exist_ok=True)

    big_dataset = _kwm_get_big_dataset(verbose=True, one_student=True)
    df_no_cache = _kwm_measure_time_on_dataset(matcher, big_dataset, verbose)
    df_no_cache.to_csv(f"_output/timings/{model_name}_times_no_cache.csv", index=False)

    df_cached = _kwm_measure_time_on_dataset(matcher, big_dataset, verbose)
    df_cached.to_csv(f"_output/timings/{model_name}_times_cached.csv", index=False)

    big_dataset[
        0
    ].text_2 += ". Этот текст был изменен, чтобы проверить скорость пересчёта"
    df_changed_vacancy = _kwm_measure_time_on_dataset(matcher, big_dataset, verbose)
    df_changed_vacancy.to_csv(
        f"_output/timings/{model_name}_times_no_cache_changed_vacancy.csv", index=False
    )

    for datum in big_dataset:
        datum.text_1 += ". Этот текст был изменен, чтобы проверить скорость пересчёта"
    df_changed_resume = _kwm_measure_time_on_dataset(matcher, big_dataset, verbose)
    df_changed_resume.to_csv(
        f"_output/timings/{model_name}_times_no_cache_changed_resume.csv", index=False
    )

    big_dataset_swapped = []
    for datum in big_dataset:
        big_dataset_swapped.append(
            Datum(
                text_1=datum.text_2,
                text_2=datum.text_1,
                title_1=datum.title_2,
                title_2=datum.title_1,
                need_match=datum.need_match,
            )
        )
    df_swapped = _kwm_measure_time_on_dataset(matcher, big_dataset_swapped, verbose)
    df_swapped.to_csv(
        f"_output/timings/{model_name}_times_no_cache_swapped.csv", index=False
    )


def get_timings_methods():
    return [
        (
            "keywords",
            KeywordDistanceMatcher(
                KwDistanceMatcherParams(
                    is_window=True, kw_saturation=True, swap_texts=True
                ),
                False,
                init_cache("memory"),
            ),
        ),
        ("gpt", GPTMatcher(GPTMatcherParams(), False, init_cache("memory"))),
        ("tfidf", _get_model("all_examples", "tfidf", init_cache("memory"))[0]),
        ("word2vec", _get_model("all_examples", "word2vec", init_cache("memory"))[0]),
        ("fastText", _get_model("all_examples", "fastText", init_cache("memory"))[0]),
        (
            "paraphrase-multilingual-MiniLM-L12-v2",
            _get_model(
                "all_examples",
                "paraphrase-multilingual-MiniLM-L12-v2",
                init_cache("memory"),
            )[0],
        ),
        (
            "rubert-base-cased",
            _get_model(
                "all_examples",
                "DeepPavlov/rubert-base-cased",
                init_cache("memory"),
            )[0],
        ),
    ]


def do_timings_total():
    timings_methods = get_timings_methods()
    for name, matcher in timings_methods[6:]:
        print(name)
        _measure_timings(matcher, name, verbose=True)


def process_timings():
    timings_methods = get_timings_methods()

    result = []

    for model_name, _ in timings_methods:
        df_no_cache = pd.read_csv(f"_output/timings/{model_name}_times_no_cache.csv")
        df_cached = pd.read_csv(f"_output/timings/{model_name}_times_cached.csv")
        df_changed_vacancy = pd.read_csv(
            f"_output/timings/{model_name}_times_no_cache_changed_vacancy.csv"
        )
        df_changed_resume = pd.read_csv(
            f"_output/timings/{model_name}_times_no_cache_changed_resume.csv"
        )
        df_swapped = pd.read_csv(
            f"_output/timings/{model_name}_times_no_cache_swapped.csv"
        )

        result.append(
            {
                "model": model_name,
                "no_cache": df_no_cache["time"].mean(),
                "no_cache_pt": df_no_cache["process_time"].mean(),
                "cache": df_cached["time"].mean(),
                "cache_pt": df_cached["process_time"].mean(),
                "changed_vacancy": df_changed_vacancy["time"].mean(),
                "changed_vacancy_pt": df_changed_vacancy["process_time"].mean(),
                "changed_resume": df_changed_resume["time"].mean(),
                "changed_resume_pt": df_changed_resume["process_time"].mean(),
                "swap": df_swapped["time"].mean(),
                "swap_pt": df_swapped["process_time"].mean(),
            }
        )
        plt.rcParams["savefig.dpi"] = 300

        fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(8, 4))
        key = 'time'
        df_no_cache[[key, "text_2_len"]].plot(
            ax=ax1,
            x="text_2_len",
            y=key,
            label="No cache",
            kind="scatter",
            logx=True,
            title=f"{key} vs vacancy length",
        )
        df_swapped[[key, "text_1_len"]].plot(
            ax=ax2,
            x="text_1_len",
            y=key,
            label="No cache",
            kind="scatter",
            logx=True,
            title=f"{key} vs resume length",
        )
        plt.tight_layout()
        for ax in [ax1, ax2]:
            ax.get_legend().set_visible(False)
            ax.grid(0.25)
            ax.set_axisbelow(True)

        fig.savefig(f"_output/timings/{model_name}_times_no_cache.png")

    df = pd.DataFrame(result)
    df.to_csv("_output/timings/timings.csv", index=False)


def run_for_mprof(model_name: str, no_cache=False):
    big_dataset = _kwm_get_big_dataset(verbose=False, one_student=True)
    if no_cache:
        cache = init_cache("dummy")
    else:
        cache = init_cache("memory")

    matcher, _ = _get_model("all_examples", model_name, cache, verbose=False)
    for datum in tqdm_v(big_dataset, verbose=True):
        matcher.match(datum.text_1, datum.text_2)
