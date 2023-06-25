import random
from dataclasses import dataclass, field
import pandas as pd
import cachetools
import gensim
import gensim.models as models
from dss_benchmark.common import *
from timeit import default_timer as timer
import numpy as np
from dss_benchmark.methods import AbstractSimilarityMethod
from dss_benchmark.common.preprocess.anmatveev.common import *
import sentence_transformers
import pymorphy2
from nltk.corpus import stopwords
from pathlib import Path
from sklearn.metrics import auc

__all__ = ['MatchManager']

round_number = 3

class MatchManager(AbstractSimilarityMethod):
    def __init__(self,
                 model_path,
                 params =None,
                 verbose=False,
                 cache: cachetools.Cache = None):
        if cache is None:
            cache = init_cache("redis")
        self._cache = cache
        self._verbose = verbose
        self._models = {}
        self._current_model = {}
        self._params = params
        if model_path:
            self.load_model(model_path)

    def preprocess_and_save_pairs(self, data_df: pd.DataFrame, text_field_1, text_field_2):
        data_df_preprocessed = data_df.copy()
        data_df_preprocessed['preprocessed_' + text_field_1] = data_df_preprocessed.apply(
            lambda row: preprocess(row[text_field_1], punctuation_marks, stop_words, morph), axis=1)
        data_df_preprocessed['preprocessed_' + text_field_2] = data_df_preprocessed.apply(
            lambda row: preprocess(row[text_field_2], punctuation_marks, stop_words, morph), axis=1)
        data_df_preprocessed = data_df_preprocessed.drop(columns=[text_field_1, text_field_2], axis=1)
        data_df_preprocessed.reset_index(drop=True, inplace=True)
        return data_df_preprocessed

    def load_model(self, model_path):
        if model_type(model_path) == "gensim":
            self._models[model_path] = models.ldamodel.LdaModel.load(model_path)
        elif model_type(model_path) == "transformer":
            self._models[model_path] = sentence_transformers.SentenceTransformer(model_path)

    def set_current_model(self, model_path):
        self._current_model['model'] = self._models[model_path]
        self._current_model['path'] = model_path

    def match(self, text_1: str, text_2: str) -> float:
        key = text_1 + text_2 + self._current_model['path']
        if self._params:
            key += str(self._params.window) + str(self._params.epochs) \
                   + str(self._params.sg) + str(self._params.min_count) + str(self._params.vector_size)
        cached = self._cache.get(key)
        value = None
        if cached:
            return cached

        if model_type(self._current_model['path']) == "gensim":
            text_1 = preprocess(text_1, punctuation_marks, stop_words, morph)
            text_2 = preprocess(text_2, punctuation_marks, stop_words, morph)
            text_1 = [w for w in text_1 if w in self._current_model['model'].wv.index_to_key]
            text_2 = [w for w in text_2 if w in self._current_model['model'].wv.index_to_key]
            value = float(round(self._current_model['model'].wv.n_similarity(text_1, text_2), round_number))
        elif model_type(self._current_model['path']) == "transformer":
            sentences_1 = sent_preprocess(text_1)
            sentences_2 = sent_preprocess(text_2)
            def comp(e):
                return e['cos_sim']

            sentences_2_embeddings = []
            for sent_2 in sentences_2:
                sentences_2_embeddings += [self._current_model['model'].encode(sent_2, convert_to_tensor=True)]

            max_sims = []
            for sent_1 in sentences_1:
                sent_1_embedding = self._current_model['model'].encode(sent_1, convert_to_tensor=True)
                sim = []
                for i in range(len(sentences_2_embeddings)):
                    sim += [{'proj': sentences_2[i],
                             'cos_sim': float(sentence_transformers.util.cos_sim(
                                 sent_1_embedding,
                                 sentences_2_embeddings[i]))
                             }]
                max_sims.append(max(sim, key=comp)['cos_sim'])
            value = float(np.round(np.mean(max_sims), round_number))
        self._cache[key] = value
        return value

    def max_f1(self, sentences_1, sentences_2, df, model_path, step=0.02):
        if sentences_1.size != sentences_2.size:
            return None
        else:
            sim = []
            self.set_current_model(model_path)
            if sentences_1.size != sentences_2.size:
                return None
            else:
                start = timer()
                model_name = Path(model_path).stem
                for i in range(sentences_1.size):
                    sim += [self.match(sentences_1[i], sentences_2[i])]
                steps, thresholds, f1_score, cutoff = max_f1_score(sim, df, step=step)
                print(f"Time of computing {model_name}: {round(timer() - start, 3)}")
                res = {}
                preds = [sim[i] >= cutoff for i in range(len(df))]
                metrics = calc_all(sim, df, cutoff)
                res["steps"] = steps
                res["thresholds"] = thresholds
                res["cutoff"] = cutoff
                res["f1-score"] = metrics["f1-score"]
                res["precision"] = metrics["precision"]
                res["recall"] = metrics["recall"]
                res["preds"] = preds
                return res

    def roc_auc(self, sentences_1, sentences_2, df, model_path, step=0.02):
        if sentences_1.size != sentences_2.size:
            return None
        else:
            sim = []
            self.set_current_model(model_path)
            if sentences_1.size != sentences_2.size:
                return None
            else:
                start = timer()
                for i in range(sentences_1.size):
                    sim += [self.match(sentences_1[i], sentences_2[i])]
                print(f"Time of computing {model_path}: {round(timer() - start, 3)}")
                res = {}
                steps, tprs, fprs, cutoff = max_diff_tpr_fpr(sim, df)
                roc_auc = auc(fprs, tprs)
                preds = [sim[i] >= cutoff for i in range(len(df))]
                model_path = Path(model_path).stem
                if "paraphrase-multilingual-MiniLM-L12-v2" == model_path:
                    model_path = "multilingual"
                res["steps"] = steps
                res["tprs"] = tprs
                res["fprs"] = fprs
                res["cutoff"] = cutoff
                res["auc"] = round(roc_auc, round_number)
                res["preds"] = preds
                return res
