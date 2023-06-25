from dss_benchmark.common import tqdm_v
from fuzzysearch import find_near_matches
from sklearn.base import BaseEstimator, ClassifierMixin
from thefuzz import fuzz

__all__ = ["CombinedRatioMatcher"]


class CombinedRatioMatcher(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        keyphrases,
        max_l_dist=9,
        near_match_norm=25,
        near_match_treshold=10,
        near_match_max_length=1500,
        verbose=False,
    ):
        self.keyphrases = keyphrases
        self.max_l_dist = max_l_dist
        self.near_match_norm = near_match_norm
        self.near_match_treshold = near_match_treshold
        self.near_match_max_length = near_match_max_length
        self.verbose = verbose

    def fit(self, X=None, y=None):
        pass

    def _get_near_match_distance(self, l, text):
        text = text[: self.near_match_max_length]
        matches = find_near_matches(l, text, max_l_dist=self.max_l_dist)
        if matches is None or len(matches) == 0:  # type: ignore
            return 0

        best_match = sorted(matches, key=lambda m: m.dist)[0]
        best_match_dist = best_match.dist
        label_len = len(l)
        return max(0, (label_len - best_match_dist) / label_len) * 100

    def _get_one_score(self, text, label):
        md = 0
        if len(label) > self.near_match_treshold:
            md = self._get_near_match_distance(label, text)
        return max([fuzz.partial_ratio(text, label), md])

    def _get_score(self, text, labels):
        return [
            self._get_one_score(text, label) for label in tqdm_v(labels, self.verbose)
        ]

    def predict(self, X):
        for x in X:
            yield self._get_score(x, self.keyphrases)
