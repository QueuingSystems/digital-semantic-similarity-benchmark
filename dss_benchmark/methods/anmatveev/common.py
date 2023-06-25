import numpy as np

__all__ = [
    "get_states",
    "max_f1_score",
    "calc_f1_score",
    "calc_all",
    "calc_tpr_fpr",
    "max_diff_tpr_fpr",
    "model_type",
]


GENSIM_MODELS = ["word2vec", "fastText"]
TRANSFORMER_MODELS = ["paraphrase-multilingual-MiniLM-L12-v2", "rubert-base-cased"]


def get_states(sim, df, match_threshold):
    (TP, FP, FN, TN) = (0, 0, 0, 0)
    # print(sim)
    for i in range(len(sim)):
        if df["need_match"][i]:
            if sim[i] >= match_threshold:
                TP += 1
            else:
                FN += 1
        else:
            if sim[i] >= match_threshold:
                FP += 1
            else:
                TN += 1

    return TP, FP, FN, TN


def max_f1_score(sim, df, step=0.02):
    score = 0
    scores = []
    f1_score = 0
    cutoff = 0
    h = step
    steps = np.linspace(0, 1, num=int(1 / h) + 1)
    steps = np.round(steps, 2)
    for i in steps:
        score = calc_f1_score(sim, df, i)
        scores.append(score)
        if score > f1_score:
            f1_score = score
            cutoff = i
    return steps, scores, f1_score, round(cutoff, 3)


def calc_f1_score(sim, df, match_threshold):
    (TP, FP, FN, TN) = get_states(sim, df, match_threshold)
    #     print(TP, FP, FN, TN)
    return round(float(2 * TP / (2 * TP + FP + FN)), 3)


def calc_all(sim, df, match_threshold):
    (TP, FP, FN, TN) = get_states(sim, df, match_threshold)
    return {
        "f1-score": round(float(2 * TP / (2 * TP + FP + FN)), 3),
        "precision": round(float(TP / (TP + FP)), 3),
        "recall": round(float(TP / (TP + FN)), 3),
    }


def calc_tpr_fpr(sim, df, match_threshold):
    (TP, FP, FN, TN) = get_states(sim, df, match_threshold)
    return {
        "tpr": round(float(TP / (TP + FN)), 3),
        "fpr": round(float(FP / (FP + TN)), 3),
    }


def max_diff_tpr_fpr(sim, df, step=0.02):
    score = 0
    diff_tpr_fpr = 0
    cutoff = 0
    h = step
    steps = np.linspace(0, 1, num=int(1 / h) + 1)
    steps = np.round(steps, 2)
    tprs = []
    fprs = []
    for i in steps:
        try:
            score = calc_tpr_fpr(sim, df, i)
            tprs.append(score["tpr"])
            fprs.append(score["fpr"])
            diff = round(float((score["tpr"] - score["fpr"])), 3)
            if diff > diff_tpr_fpr:
                diff_tpr_fpr = diff
                cutoff = i
        except ZeroDivisionError:
            print("ZeroDiv")
            pass
    return steps, tprs, fprs, cutoff


def model_type(model_path):
    for model in GENSIM_MODELS:
        if model.lower() in model_path.lower():
            return "gensim"
    for model in TRANSFORMER_MODELS:
        if model.lower() in model_path.lower():
            return "transformer"
    return None
