import itertools
from collections import deque

from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = ["KwWindowExtractor"]


class KwWindowExtractor(BaseEstimator, ClassifierMixin):
    def __init__(self, model, window_size=250, window_delta=50):
        self.window_size = window_size
        self.window_delta = window_delta
        self.model = model

    def fit(self, X=None, Y=None):
        self.model.fit(X, Y)

    def transform(self, X):
        return self.predict(X)

    def _get_windows(self, text: str):
        result = []
        sentences = text.split(".")
        current_thing = deque()
        current_thing_len = 0
        for sentence in sentences:
            if current_thing_len > self.window_size:
                result.append(".".join(current_thing))

                while current_thing_len > self.window_delta:
                    elem = current_thing.popleft()
                    current_thing_len -= len(elem)

            current_thing.append(sentence)
            current_thing_len += len(sentence)
        if current_thing_len > 0:
            result.append(".".join(current_thing))
        return result

    def predict(self, X):
        for x in X:
            windows = self._get_windows(x)
            result = list(
                itertools.chain(*[next(iter(self.model.predict([w]))) for w in windows])
            )
            yield result
