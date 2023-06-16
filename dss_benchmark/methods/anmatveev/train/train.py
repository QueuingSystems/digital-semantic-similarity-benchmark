from dataclasses import dataclass, field
from dss_benchmark.common import EmptyMapping
import pandas as pd
import cachetools
import gensim
from dss_benchmark.common import *
from timeit import default_timer as timer

@dataclass
class TrainModelParams:
    texts:str = field(
         metadata={"help": "Обучающий набор (путь к файлу)"}
    )
    vector_size:int = field(
        default=100,
        metadata={"help": "Размерность векторов слов"}
    )
    window:int = field(
        default=5,
        metadata={"help": "Размер окна (максимальное расстояние между текущим и прогнозируемым словом в предложении)"}
    )
    min_count:int = field(
        default=5,
        metadata={"help": "Игнорирует все слова с общей частотой ниже этой."}
    )
    max_vocab_size:any = field(
        default=None,
        metadata={"help": "Ограничивает оперативную память при построении словаря; None - нет ограничений"}
    )
    alpha:float = field(
        default=0.025,
        metadata={"help": "Начальная скорость обучения"}
    )
    sample:float = field(
        default=0.001,
        metadata={"help": "Порог для настройки того, какие высокочастотные слова будут уменьшаться"}
    )
    seed:int = field(
        default=1,
        metadata={"help": "Seed для генератора случайных чисел"}
    )
    workers:int = field(
        default=3,
        metadata={"help": "Рабочие потоки для обучения модели"}
    )
    min_alpha:float = field(
        default=0.0001,
        metadata={"help": "Скорость обучения будет линейно падать до min_alpha"}
    )
    sg:int = field(
        default= 0,
        metadata={"help": "Алгоритм обучения: 1 для скип-граммы; иначе CBOW"}
    )
    hs:int = field(
        default= 0,
        metadata={"help": "Если 1 - иерархический softmax. Если 0, и negative не 0 - negative sampling"}
    )
    negative:int = field(
        default=5,
        metadata={"help": "Если > 0, будет использоваться обучающая выборка"}
    )
    ns_exponent:float = field(
        default=0.75,
        metadata={"help": "Показатель степени, используемый для формирования распределения отрицательной выборки."}
    )
    cbow_mean:int = field(
        default=1,
        metadata={"help": "0 - сумма векторов контекстных слов, 1 - используется среднее (только в CВOW)"}
    )
    epochs:int = field(
        default=5,
        metadata={"help": "Количество итераций по корпусу"}
    )
    sorted_vocab:int = field(
        default=1,
        metadata={"help": "если 1, отсортирует словарь по частоте убывания, прежде чем назначать индексы"}
    )
    batch_words:int = field(
        default=10000,
        metadata={"help": "Целевой размер (в словах) для пакетов примеров"}
    )
    compute_loss:bool = field(
        default=False,
        metadata={"help": "Если True, вычисляет и сохраняет значение потерь"}
    )
    max_final_vocab:int = field(
        default=None,
        metadata={"help": "Ограничивает словарь до целевого размера словаря"}
    )
    shrink_windows:bool = field(
        default=True,
        metadata={"help": "Если установлено значение True, эффективный размер окна равномерно выбирается"}
    )


class TrainModelManager:
    def __init__(self, 
                 params=TrainModelParams, 
                 verbose=False,
                 cache: cachetools.Cache = None):
        self.params = params
        self.params = params
        if cache is None:
            cache = EmptyMapping()
        self._cache = cache
        self._verbose = verbose
    
    def preprocess_and_save(self, data_df: pd.DataFrame, path, text_field='text') -> pd.DataFrame:
        # for preprocessing dataset. Use it only in critical cases cause it's too slow on big datasets
        data_df['preprocessed_' + text_field] = data_df.apply(
            lambda row: preprocess(row[text_field], punctuation_marks, stop_words, morph), axis=1)
        data_df_preprocessed = data_df.copy()
        data_df_preprocessed = data_df_preprocessed.drop(columns=[text_field], axis=1)
        data_df_preprocessed.reset_index(drop=True, inplace=True)
        if path is not None:
            data_df_preprocessed.to_json(path)
        return data_df_preprocessed
    
    def train(self,  model="word2vec", model_path="models/"):
        print(self.params, model)
        if model == "word2vec":
            train_part = pd.read_json(self.params.texts)['preprocessed_text']
            start = timer()
            self.model = gensim.models.Word2Vec(sentences=train_part,
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
                                                shrink_windows=self.params.shrink_windows)

            print(f'Training {model} time: {round(timer() - start, 3)} secs')
            self.model.save(model_path + model)
        # elif model == "fastText":
        #     train_part = data_df['preprocessed_texts']
        #     self.model = gensim.models.FastText(sentences=train_part, min_count=5, vector_size=50, epochs=10)
        #     self.model.save(model_path + model)
        # return

    

