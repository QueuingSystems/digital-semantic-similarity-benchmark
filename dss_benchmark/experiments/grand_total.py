from typing import List

import pandas as pd
from dss_benchmark.common import init_cache, tqdm_v
from dss_benchmark.experiments.common import (
    ResultDatum,
    confusion_matrix,
    f1_score,
    load_dataset,
    process_auprc,
    process_f1_score,
    process_roc_auc,
)
from dss_benchmark.experiments.keyword_matching import kwm_match_parallel
from dss_benchmark.methods.chatgpt import GPTMatcher, GPTMatcherParams
from dss_benchmark.methods.keyword_matching import (
    KeywordDistanceMatcher,
    KwDistanceMatcherParams,
)
from dss_benchmark.methods.tfidf.tfidf import TfIdfMatcher, TfIdfMatcherParams

__all__ = ["do_grand_total"]

MODELS = {"keywords": KeywordDistanceMatcher, "gpt": GPTMatcher, "tfidf": TfIdfMatcher}

GPT_SHOTS = {
    "all_examples": [(13, 100), (28, 100)],
    "studentor_partner": [(18, 90), (110, 100)],
    "dataset_v6_r30": [],
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
        )
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
        )
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
        )
    },
}


def _do_match(dataset_name: str, model_name: str, cache, verbose=False):
    params = BEST_PARAMS[dataset_name][model_name]
    dataset = load_dataset(dataset_name)
    if model_name == "keywords":
        return kwm_match_parallel(params, 0, dataset, verbose=verbose)
    Model = MODELS[model_name]

    result: List[ResultDatum] = []
    matcher = Model(params, cache=cache)

    if model_name == "gpt":
        shots = GPT_SHOTS.get(dataset_name, [])
        for idx, val in shots:
            matcher.add_shot(
                dataset[idx].text_1, dataset[idx].text_2, int(val / 10), tokens=2000
            )

    for datum in tqdm_v(dataset, verbose=verbose):
        value = matcher.match(datum.text_1, datum.text_2)
        result.append(ResultDatum(datum=datum, value=value, match=value >= 0))
    return result


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


def do_grand_total():
    results = []
    cache = init_cache()
    dataset_names = list(BEST_PARAMS.keys())
    for dataset_name in dataset_names:
        print(f"Dataset: {dataset_name}")
        for model_name in MODELS.keys():
            print(f"Model: {model_name}")
            match_result = _do_match(dataset_name, model_name, cache, verbose=True)
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
