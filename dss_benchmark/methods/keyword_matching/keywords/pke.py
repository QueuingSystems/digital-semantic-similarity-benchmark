from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = ["PkeExtractor"]


class PkeExtractor(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        model,
        n=10,
        threshold=0,
        load_kwargs=None,
        candidate_selection_kwargs=None,
        candidate_weighting_kwargs=None,
    ):
        self.n = n
        self.model = model
        self.threshold = threshold
        if candidate_selection_kwargs is None:
            self.candidate_selection_kwargs = {}
        else:
            self.candidate_selection_kwargs = candidate_selection_kwargs
        if candidate_weighting_kwargs is None:
            self.candidate_weighting_kwargs = {}
        else:
            self.candidate_weighting_kwargs = candidate_weighting_kwargs
        if load_kwargs is None:
            self.load_kwargs = {}
        else:
            self.load_kwargs = load_kwargs

    def fit(self, X=None, Y=None):
        pass

    def predict(self, X=None):
        for x in X:
            extractor = self.model()
            extractor.load_document(input=str(x), **self.load_kwargs)
            extractor.candidate_selection(**self.candidate_selection_kwargs)
            extractor.candidate_weighting(**self.candidate_weighting_kwargs)
            n_best = extractor.get_n_best(self.n, self.threshold)
            yield [{"value": v, "certainty": s} for v, s in n_best]
