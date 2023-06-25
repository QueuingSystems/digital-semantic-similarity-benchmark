from thefuzz import fuzz
from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = ["KeywordsExcludeBlacklist"]


class KeywordsExcludeBlacklist(BaseEstimator, ClassifierMixin):
    def __init__(self, blacklist, min_match=92):
        self.blacklist = blacklist
        self.min_match = min_match

    def _kw_is_bad(self, kw: str):
        for item in self.blacklist:
            if fuzz.partial_ratio(item, kw) >= self.min_match:
                return True
        return False

    def fit(self, X=None, Y=None):
        pass

    def transform(self, X):
        return self.predict(X)

    def predict(self, X):
        for x in X:
            yield [d for d in x if not self._kw_is_bad(d["value"])]
