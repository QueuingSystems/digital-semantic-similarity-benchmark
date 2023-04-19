from dataclasses import dataclass, field

import nltk
import numpy as np
import pke
import spacy
from dss_benchmark.common.preprocess import TextNormalizer, TextSnowballStemmer
from dss_benchmark.methods import AbstractSimilarityMethod
from dss_benchmark.methods.keyword_matching.distance import CombinedRatioMatcher
from dss_benchmark.methods.keyword_matching.keywords import (
    KeywordsClusterNew,
    KeywordsExcludeBlacklist,
    KeywordsExcludeRussian,
    KwWindowExtractor,
    MultipleExtractor,
    PkeExtractor,
)
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

__all__ = ["KeywordDistanceMatcher", "KwDistanceMatcherParams"]

nlp = spacy.load("ru_core_news_sm")

BLACKLIST = ["умение", "знание", "опыт", "работа", "стажировка", "практика", "студент"]

yake_kws_best = PkeExtractor(
    pke.unsupervised.YAKE,
    load_kwargs={
        "spacy_model": nlp,
        "stoplist": nltk.corpus.stopwords.words("russian"),
        "normalization": None,
    },
    candidate_selection_kwargs={
        "n": 2,
        # 'stoplist': nltk.corpus.stopwords.words('russian')
    },
    candidate_weighting_kwargs={
        "window": 3,
        # 'stoplist': nltk.corpus.stopwords.words('russian'),
        "use_stems": False,
    },
)

positionrank_kws_best = PkeExtractor(
    pke.unsupervised.PositionRank,
    load_kwargs={
        "spacy_model": nlp,
        "stoplist": nltk.corpus.stopwords.words("russian"),
        "normalization": None,
    },
    candidate_selection_kwargs={
        "maximum_word_number": 3,
        "grammar": "NP:{<ADJ>*<NOUN|PROPN>+}",
    },
    candidate_weighting_kwargs={
        "window": 3,
        "pos": {"NOUN", "PROPN"},
    },
)

singlerank_kws_best = PkeExtractor(
    pke.unsupervised.SingleRank,
    load_kwargs={
        "spacy_model": nlp,
        "stoplist": nltk.corpus.stopwords.words("russian"),
        "normalization": None,
    },
    candidate_selection_kwargs={"pos": {"NOUN", "PROPN"}},
    candidate_weighting_kwargs={"window": 3, "pos": {"NOUN", "PROPN"}},
)


@dataclass
class KwDistanceMatcherParams:
    is_window: bool = field(
        default=False,
        metadata={"help": "Использовать окно для извлечения ключевых слов"},
    )
    window_size: int = field(
        default=170,
        metadata={"help": "Размер окна"},
    )
    window_delta: int = field(
        default=50,
        metadata={"help": "Шаг окна"},
    )
    saturate_match: bool = field(
        default=False,
        metadata={"help": "Сопоставление с насыщением"},
    )
    saturate_value: int = field(
        default=6,
        metadata={"help": "Столько раз найдется ключевое слово -> 100% совпадаение"},
    )
    dbscan_eps: float = field(
        default=0.95,
        metadata={"help": "Параметр eps для DBSCAN"},
    )
    kw_cutoff: float = field(
        default=0.725,
        metadata={"help": "Минимальная уверенность в ключевом слове"},
    )
    swap_texts: bool = field(
        default=False,
        metadata={"help": "Поменять местами тексты"},
    )


class KeywordDistanceMatcher(AbstractSimilarityMethod):
    def __init__(self, params: KwDistanceMatcherParams):
        self.params = params

        kw_extractor = MultipleExtractor(
            [
                (yake_kws_best, "YAKE"),
                (positionrank_kws_best, "PositionRank"),
                (singlerank_kws_best, "SingleRank"),
            ]
        )
        if params.is_window:
            extractor = KwWindowExtractor(
                kw_extractor,
                window_size=params.window_size,
                window_delta=params.window_delta,
            )
        else:
            extractor = kw_extractor

        russian = KeywordsExcludeRussian()
        blacklist = KeywordsExcludeBlacklist(BLACKLIST)

        cluster = KeywordsClusterNew(
            Pipeline(
                [
                    ("normalizer", TextNormalizer()),
                    ("stemmer", TextSnowballStemmer()),
                    ("vectorizer", TfidfVectorizer(ngram_range=(1, 3))),
                    ("cluster", DBSCAN(min_samples=1, eps=params.dbscan_eps)),
                ]
            ),
            source_selection=["YAKE", "PositionRank", "SingleRank"],
        )

        self.kw_pipeline = Pipeline(
            [
                ("extract", extractor),
                ("russian", russian),
                ("blacklist", blacklist),
                ("cluster", cluster),
            ]
        )

    def _extract_keywords(self, text: str):
        keywords = list(self.kw_pipeline.transform([text]))[0]
        return keywords

    def _match(self, text: str, keywords):
        if len(keywords) == 0:
            return 0.0
        matcher = CombinedRatioMatcher(keywords)
        results = list(matcher.predict([text.lower()]))[0]
        for i in range(len(results)):
            if results[i] < self.params.kw_cutoff * 100:
                results[i] = 0
        return np.mean(results)

    def _match_saturated(self, text: str, keywords):
        if len(keywords) == 0:
            return 0.0
        matcher = CombinedRatioMatcher(keywords)
        results = list(matcher.predict([text.lower()]))[0]
        results = sorted(results, reverse=True)[: self.params.kw_saturation]
        for i in range(len(results)):
            if results[i] < self.params.kw_cutoff * 100:
                results[i] = 0
        return np.mean(results)

    def match(self, text_1: str, text_2: str) -> float:
        if self.params.swap_texts:
            text_1, text_2 = text_2, text_1

        # TODO кэширование ключевых слов и результатов
        kw_1 = self._extract_keywords(text_1)

        if self.params.saturate_match:
            return self._match_saturated(text_2, kw_1)
        else:
            return self._match(text_2, kw_1)
