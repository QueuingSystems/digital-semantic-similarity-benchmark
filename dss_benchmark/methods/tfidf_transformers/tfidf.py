from dataclasses import dataclass, field
from dss_benchmark.common import EmptyMapping
from dss_benchmark.methods.tfidf_transformers import Text_preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
import cachetools
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import os
import pickle
from dss_benchmark.methods import AbstractSimilarityMethod
@dataclass
class TfIdfParams:
    train_data:str = field(
         default='all_examples',
         metadata={"help": "Путь до датасета, на основе которого нужно выполнить обучение"}
    )
    min_ngram:int = field(
        default=1,
        metadata={"help": "Минимальный размер n-граммы"}
    )
    max_ngram:int = field(
        default=3,
        metadata={"help": "Максимальный размер n-граммы"}
    )
    binary: bool = field(
        default=False,
        metadata={"help": "Двоичный tf-idf"}
    )
    sublinear_tf: bool = field(
        default=False,
        metadata={"help": "Сублинейное масштабирование (tf=1+log(tf))"}
    )



class TfIdf(AbstractSimilarityMethod):
    def __init__(
        self,
        params: TfIdfParams,
        verbose=False,
        cache: cachetools.Cache = None,
    ):
        self.params = params
        if cache is None:
            cache = EmptyMapping()
        self._cache = cache
        self._verbose = verbose
        self.names = {
            "rpd_dataset" : "rpd_dataset.json", 
            "all_examples" : "all_examples.json", 
            "studentor_partner" : "studentor_partner.csv", 
            'dataset_v6_r30' : 'dataset_v6_r30.csv',}
        self.lema = Text_preprocessing()
        os.makedirs('models/', exist_ok=True)
        self.info = 'tfidf (train dataset {} ngram ({},{}) binary {} sublinear {})'.format(
            params.train_data, self.params.min_ngram, self.params.max_ngram, 
            str(self.params.binary), str(self.params.sublinear_tf))
        self.name_model = 'models/tfidf_train_{}_ngram_{}_{}_bin_{}_sublinear_{}.pickle'.format(
            self.params.train_data, self.params.min_ngram, self.params.max_ngram, 
            self.params.binary, self.params.sublinear_tf)
        if not os.path.exists(self.name_model):
            self.tfidf = TfidfVectorizer(
                ngram_range=(self.params.min_ngram, self.params.max_ngram), 
                binary=self.params.binary,
                sublinear_tf=self.params.sublinear_tf
            )
            bow_rep_tfidf = self.tfidf.fit_transform(self.get_data(self.params.train_data))
            pickle.dump(self.tfidf, open(self.name_model, "wb"))
        else:
            self.tfidf = pickle.load(open(self.name_model, "rb"))
        

    def match(self, text_1: str, text_2: str) -> float:
        a = self.get_transform_data(text_1)
        b = self.get_transform_data(text_2)
        if norm(a) * norm(b) == 0:
            return 0
        return dot(a, b) / (norm(a) * norm(b))*100

    def get_data(self, name:str):
        name = self.conver_name(name)
        if not os.path.exists('data/prep_'+name):
            self.proccess_data_set(name)
        df = self.read_data(name)
        field = self.get_conf_data(name)
        data = [*df[field[0]].to_numpy(), *df[field[1]].to_numpy()]
        return data

    def proccess_data_set(self, name:str):
        format = name.split('.')[-1]
        df = self.read_data(name)
        field = self.get_conf_data(name)
        for col in field:
            df[col] = df[col].apply(lambda x : self.lema.preprocess_text(x))
        if format == 'json':
            df.to_json('data/prep_'+name)
        elif format == 'csv':
            df.to_csv('data/prep_'+name)

    def get_transform_data(self, doc):
        doc = self.lema.preprocess_text(doc)
        return self.tfidf.transform([doc]).toarray()[0]
    
    def get_info(self):
        return self.info
    
    def get_conf_data(self, name):
        if name == 'all_examples.json' or name == 'all_examples.json':
            return ['text_rp', 'text_proj']
        elif name == 'studentor_partner.csv':
            return ['text_2', 'text_2']
        elif name == 'dataset_v6_r30.csv':
            return ['resume_lem', 'vacancy_lem']
        else:
            return None 
    
    def read_data(self, name):
        format = name.split('.')[-1]
        dataset = 'data/'+name
        if format == 'json':
            return pd.read_json(dataset)
        elif format == 'csv':
            return pd.read_csv(dataset)
        else:
            return None
    
    def conver_name(self, name):
        return self.names[name]
