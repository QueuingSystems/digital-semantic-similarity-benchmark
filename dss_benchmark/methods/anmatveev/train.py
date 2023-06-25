from dataclasses import dataclass, field
from timeit import default_timer as timer

import cachetools
import gensim
import pandas as pd
from dss_benchmark.common import EmptyMapping
from dss_benchmark.common.preprocess import preprocess

__all__ = ["ANMTrainModelParams", "ANMTrainModelManager"]


@dataclass
class ANMTrainModelParams:
    texts: str = field(
        default=None, metadata={"help": "Обучающий набор (путь к файлу)"}
    )
    vector_size: int = field(
        default=100, metadata={"help": "Размерность векторов слов"}
    )
    window: int = field(
        default=5,
        metadata={
            "help": "Размер окна (максимальное расстояние между текущим и прогнозируемым словом в предложении)"
        },
    )
    min_count: int = field(
        default=5, metadata={"help": "Игнорирует все слова с общей частотой ниже этой."}
    )
    max_vocab_size: any = field(
        default=None,
        metadata={
            "help": "Ограничивает оперативную память при построении словаря; None - нет ограничений"
        },
    )
    word_ngrams: int = field(
        default=1,
        metadata={
            "help": "Максимальная длина словарной n-граммы, но Gensim поддерживает только n-грамму длины 1"
        },
    )
    alpha: float = field(
        default=0.025, metadata={"help": "Начальная скорость обучения"}
    )
    sample: float = field(
        default=0.001,
        metadata={
            "help": "Порог для настройки того, какие высокочастотные слова будут уменьшаться"
        },
    )
    seed: int = field(
        default=1, metadata={"help": "Seed для генератора случайных чисел"}
    )
    workers: int = field(
        default=8, metadata={"help": "Рабочие потоки для обучения модели"}
    )
    min_alpha: float = field(
        default=0.0001,
        metadata={"help": "Скорость обучения будет линейно падать до min_alpha"},
    )
    sg: int = field(
        default=0, metadata={"help": "Алгоритм обучения: 1 для скип-граммы; иначе CBOW"}
    )
    hs: int = field(
        default=0,
        metadata={
            "help": "Если 1 - иерархический softmax. Если 0, и negative не 0 - negative sampling"
        },
    )
    negative: int = field(
        default=5, metadata={"help": "Если > 0, будет использоваться обучающая выборка"}
    )
    ns_exponent: float = field(
        default=0.75,
        metadata={
            "help": "Показатель степени, используемый для формирования распределения отрицательной выборки."
        },
    )
    cbow_mean: int = field(
        default=1,
        metadata={
            "help": "0 - сумма векторов контекстных слов, 1 - используется среднее (только в CВOW)"
        },
    )
    epochs: int = field(default=5, metadata={"help": "Количество итераций по корпусу"})
    sorted_vocab: int = field(
        default=1,
        metadata={
            "help": "если 1, отсортирует словарь по частоте убывания, прежде чем назначать индексы"
        },
    )
    batch_words: int = field(
        default=10000,
        metadata={"help": "Целевой размер (в словах) для пакетов примеров"},
    )
    compute_loss: bool = field(
        default=False,
        metadata={"help": "Если True, вычисляет и сохраняет значение потерь"},
    )
    max_final_vocab: int = field(
        default=None,
        metadata={"help": "Ограничивает словарь до целевого размера словаря"},
    )
    shrink_windows: bool = field(
        default=True,
        metadata={
            "help": "Если установлено значение True, эффективный размер окна равномерно выбирается"
        },
    )
    min_n: int = field(
        default=3, metadata={"help": "Минимальная длина символьных n-грамм"}
    )
    max_n: int = field(
        default=6,
        metadata={
            "help": "Максимальная длина n-грамм, которые будут использоваться для обучения пр-ю слов"
        },
    )
    bucket: int = field(
        default=2000000,
        metadata={
            "help": "N-граммы хэшируются в фикс. число buckets для ограничения памяти"
        },
    )


class ANMTrainModelManager:
    def __init__(
        self, params=ANMTrainModelParams, verbose=False, cache: cachetools.Cache = None
    ):
        self.params = params
        if cache is None:
            cache = EmptyMapping()
        self._cache = cache
        self._verbose = verbose

    def preprocess_and_save(
        self, data_df: pd.DataFrame, path, text_field="text"
    ) -> pd.DataFrame:
        # for preprocessing dataset. Use it only in critical cases cause it's too slow on big datasets
        data_df["preprocessed_" + text_field] = data_df.apply(
            lambda row: preprocess(row[text_field]),
            axis=1,
        )
        data_df_preprocessed = data_df.copy()
        data_df_preprocessed = data_df_preprocessed.drop(columns=[text_field], axis=1)
        data_df_preprocessed.reset_index(drop=True, inplace=True)
        if path is not None:
            data_df_preprocessed.to_json(path)
        return data_df_preprocessed

    def train(self, model="word2vec", model_path="models/"):
        print(self.params, model)
        start = timer()
        train_part = pd.read_json(self.params.texts)["preprocessed_text"]
        if model == "word2vec":
            self.model = gensim.models.Word2Vec(
                sentences=train_part,
                vector_size=self.params.vector_size,
                window=self.params.window,
                min_count=self.params.min_count,
                max_vocab_size=self.params.max_vocab_size,
                alpha=self.params.alpha,
                sample=self.params.sample,
                seed=self.params.seed,
                workers=self.params.workers,
                min_alpha=self.params.min_alpha,
                sg=self.params.sg,
                hs=self.params.hs,
                negative=self.params.negative,
                ns_exponent=self.params.ns_exponent,
                cbow_mean=self.params.cbow_mean,
                epochs=self.params.epochs,
                sorted_vocab=self.params.sorted_vocab,
                batch_words=self.params.batch_words,
                compute_loss=self.params.compute_loss,
                max_final_vocab=self.params.max_final_vocab,
                shrink_windows=self.params.shrink_windows,
            )

        elif model == "fastText":
            self.model = gensim.models.FastText(
                sentences=train_part,
                sg=self.params.sg,
                hs=self.params.hs,
                vector_size=self.params.vector_size,
                alpha=self.params.alpha,
                window=self.params.window,
                min_count=self.params.min_count,
                max_vocab_size=self.params.max_vocab_size,
                ns_exponent=self.params.ns_exponent,
                cbow_mean=self.params.cbow_mean,
                epochs=self.params.epochs,
                min_n=self.params.min_n,
                max_n=self.params.max_n,
                sorted_vocab=self.params.sorted_vocab,
                bucket=self.params.bucket,
                batch_words=self.params.batch_words,
                max_final_vocab=self.params.max_final_vocab,
                shrink_windows=self.params.shrink_windows,
            )
        print(f"Training {model} time: {round(timer() - start, 3)} secs")
        self.model.save(model_path)
