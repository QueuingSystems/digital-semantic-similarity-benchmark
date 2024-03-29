from dataclasses import dataclass, field

import cachetools
import sentence_transformers
from dss_benchmark.common import EmptyMapping
from dss_benchmark.common.preprocess import TextPreprocessor1

__all__ = ["TransformersParams", "Transformers"]


@dataclass
class TransformersParams:
    model: str = field(
        default="symanto/sn-xlm-roberta-base-snli-mnli-anli-xnli",
        metadata={"help": "Название модели"},
    )


class Transformers:
    def __init__(
        self,
        params: TransformersParams,
        verbose=False,
        cache: cachetools.Cache = None,
    ):
        self.params = params
        if cache is None:
            cache = EmptyMapping()
        self._cache = cache
        self._verbose = verbose
        self.lema = TextPreprocessor1()
        self.model = sentence_transformers.SentenceTransformer(self.params.model)
        self.info = "Transfomers {}".format(self.model)

    def match(self, text_1: str, text_2: str) -> float:
        embeddings = self.model.encode(
            [self.lema.preprocess_text(text_1), self.lema.preprocess_text(text_2)]
        )
        return (
            sentence_transformers.util.cos_sim(a=embeddings[0], b=embeddings[1]).item()
            * 100
        )

    def get_info(self):
        return self.info
