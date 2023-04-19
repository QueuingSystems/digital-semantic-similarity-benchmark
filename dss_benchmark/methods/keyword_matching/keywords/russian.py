import nltk
import pymorphy2
from pymorphy2.tagset import OpencorporaTag
from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = ["KeywordsExcludeRussian"]

BAD_SINGLE = [
    # 'NOUN', # имя существительное
    "ADJF",  # имя прилагательное (полное)
    "ADJS",  # имя прилагательное (краткое)
    "COMP",  # компаратив
    "VERB",  # глагол (личная форма)
    "INFN",  # глагол (инфинитив)
    "PRTF",  # причастие (полное)
    "PRTS",  # причастие (краткое)
    "GRND",  # деепричастие
    "NUMR",  # числительное
    "ADVB",  # наречие
    "NPRO",  # местоимение-существительное
    "PRED",  # предикатив
    "PREP",  # предлог
    "CONJ",  # союз
    "PRCL",  # частица
    "INTJ",  # междометие
]


class KeywordsExcludeRussian(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()

    def _kw_is_bad(self, kw):
        tokens = nltk.tokenize.word_tokenize(kw)
        data = [self.morph.parse(token)[0] for token in tokens]

        if len(data) == 1:
            if data[0].tag.POS in BAD_SINGLE:
                return True

        is_bad = True
        is_verb = False
        for datum in data:
            if datum.tag.POS == "NOUN" or datum.tag == OpencorporaTag("LATN"):
                is_bad = False
            elif datum.tag.POS == "VERB":
                is_verb = True
        return is_bad or is_verb

    def fit(self, X=None, Y=None):
        pass

    def transform(self, X):
        return self.predict(X)

    def predict(self, X):
        for x in X:
            yield [d for d in x if not self._kw_is_bad(d["value"])]
