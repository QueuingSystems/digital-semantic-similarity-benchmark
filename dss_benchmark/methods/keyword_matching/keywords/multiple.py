from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = ["MultipleExtractor"]


class MultipleExtractor(BaseEstimator, ClassifierMixin):
    def __init__(self, models):
        self.models = models

    def fit(self, X=None, Y=None):
        [m[0].fit(X, Y) for m in self.models]

    def transform(self, X):
        return self.predict(X)

    def predict(self, X):
        for x in X:
            model_results = [next(iter(m[0].predict([x]))) for m in self.models]
            result = []
            for kws, m in zip(model_results, self.models):
                for kw in kws:
                    if isinstance(kw, str):
                        result.append({"value": kw, "source": m[1]})
                    else:
                        result.append({**kw, "source": m[1]})
            result = sorted(result, key=lambda v: v.get("certainty", 0), reverse=True)
            yield result
