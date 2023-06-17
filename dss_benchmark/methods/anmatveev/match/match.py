from dataclasses import dataclass, field
import pandas as pd
import cachetools
import gensim
import gensim.models as models
from dss_benchmark.common import *
from timeit import default_timer as timer
import numpy as np
from dss_benchmark.common.preprocess.anmatveev.common import *
import matplotlib.pyplot as plt
import pymorphy2
from nltk.corpus import stopwords
from pathlib import Path

image_path = "images/"
fontsize = 18
plt.rcParams.update({'font.size': fontsize})
plt.rcParams['figure.dpi'] = 300
punctuation_marks = ['!', ',', '(', ')', ';', ':', '-', '?', '.', '..', '...', "\"", "/", "\`\`", "»", "«"]
stop_words = stopwords.words("russian")
morph = pymorphy2.MorphAnalyzer()
model_types = []
redis_port = 0
redis_host = '-'
line_thickness = 3


class MatchManager:
    def __init__(self,
                 verbose=False,
                 cache: cachetools.Cache = None):
        if cache is None:
            cache = EmptyMapping()
        self._cache = cache
        self._verbose = verbose
        self.models = {}

    def preprocess_and_save_pairs(self, data_df: pd.DataFrame, text_field_1, text_field_2):
        data_df_preprocessed = data_df.copy()
        data_df_preprocessed['preprocessed_' + text_field_1] = data_df_preprocessed.apply(
            lambda row: preprocess(row[text_field_1], punctuation_marks, stop_words, morph), axis=1)
        data_df_preprocessed['preprocessed_' + text_field_2] = data_df_preprocessed.apply(
            lambda row: preprocess(row[text_field_2], punctuation_marks, stop_words, morph), axis=1)
        data_df_preprocessed = data_df_preprocessed.drop(columns=[text_field_1, text_field_2], axis=1)
        data_df_preprocessed.reset_index(drop=True, inplace=True)
        return data_df_preprocessed

    def set_model(self, model_path):
        self.models[model_path] = models.ldamodel.LdaModel.load(model_path)

    def predict_sim_gensim(self, sentences_1, sentences_2, model_path):
        self.model = self.models[model_path]
        if sentences_1.size != sentences_2.size:
            return None
        else:
            if self.model is not None:
                sentences_sim = np.zeros(sentences_1.size)
                sz = sentences_1.size
                sentences_1_words, sentences_2_words = None, None
                for i in range(sz):
                    if 'word2vec' in model_path:
                        sentences_1_words = [w for w in sentences_1[i] if w in self.model.wv.index_to_key]
                        sentences_2_words = [w for w in sentences_2[i] if w in self.model.wv.index_to_key]
                    elif 'fastText' in model_path:
                        sentences_1_words = sentences_1[i]
                        sentences_2_words = sentences_2[i]

                    sim = self.model.wv.n_similarity(sentences_1_words, sentences_2_words)
                    sentences_sim[i] = sim

                return list(sentences_sim)
            else:
                return None

    def max_f1(self, sentences_1, sentences_2, df, model_path, step=0.02):
        if sentences_1.size != sentences_2.size:
            return None
        else:
            threshold = 0
            thresholds = []
            f1_score = 0
            h = step
            steps = np.linspace(0, 1, num=int(1 / h))
            steps = np.round(steps, 2)
            sim = []
            try:
                sim = self.predict_sim_gensim(sentences_1, sentences_2, model_path)
            except Exception:
                pass
                # for i in range(len(sentences_1)):
                #     sim += [self.predict_transfomer_two_texts(sentences_1[i], sentences_2[i], model_name=model_name)]

            steps, thresholds, f1_score, cutoff = max_f1_score(sim, df, step=0.02)
            fig = plt.figure(figsize=(7, 6))
            plt.grid(True)
            start = timer()
            model_name = Path(model_path).stem
            if "paraphrase-multilingual-MiniLM-L12-v2" in model_path:
                model_name = "multilingual"
            plt.title(f"Maximization: {model_name}")
            plt.xlabel("Cutoff", fontsize=fontsize)
            plt.ylabel("F1-score", fontsize=fontsize)
            plt.plot(steps, thresholds, label="F1-score(cutoff)", linewidth=line_thickness)
            plt.plot(cutoff, f1_score, "r*", label="Max F1-score" )
            plt.annotate(f'({cutoff}, {f1_score})', (cutoff - 0.06, f1_score + 0.01), fontsize=fontsize - 6)
            plt.legend(loc="best")
            imname = "Maximization-F1-score-" + model_name + ".png"
            plt.savefig(image_path + imname)
            print(image_path + imname)
            res = {}
            preds = [sim[i] >= cutoff for i in range(len(df))]
            metrics = calc_all(sim, df, cutoff)
            res.setdefault("cutoff", cutoff)
            res.setdefault("f1-score", metrics["f1-score"])
            res.setdefault("precision", metrics["precision"])
            res.setdefault("recall", metrics["recall"])
            res.setdefault("sim", str(preds))
            print(f"Time of computing {model_name}: {timer() - start}")
            return res