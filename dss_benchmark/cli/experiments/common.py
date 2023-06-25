from typing import List

import pandas as pd
from dss_benchmark.experiments import (
    Datum,
    ResultDatum,
    confusion_matrix,
    f1_score,
    process_auprc,
    process_f1_score,
    process_roc_auc,
)

__all__ = ["print_results"]


def print_results(results: List[ResultDatum], dataset: List[Datum]):
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

    precision, recall = 0, 0
    if tp + fp > 0:
        precision = tp / (tp + fp)
    if tp + fn > 0:
        recall = tp / (tp + fn)

    print(f"TP: {str(tp):3s} FP: {str(fp):3s}")
    print(f"FN: {str(fn):3s} TN: {str(tn):3s}")
    print(f"F1: {f1:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")

    _, _, auprc, auprc_cutoff, auprc_f1 = process_auprc(dataset, results)
    print(f"AUPRC: {auprc:.4f}, cutoff: {auprc_cutoff:.4f}, F1:{auprc_f1:.4f}")

    _, _, auc_cutoff, auc = process_roc_auc(dataset, results)
    print(f"AUC: {auc:.4f}, cutoff: {auc_cutoff:.4f}")
