from dss_benchmark.common import tqdm_v
from dss_benchmark.common.cache import EmptyMapping

from .combined_ratio import CombinedRatioMatcher

__all__ = ["CombinedRatioMatcherCache"]


class CombinedRatioMatcherCache(CombinedRatioMatcher):
    def __init__(self, *args, cache=EmptyMapping, **kwargs):
        super().__init__(*args, **kwargs)
        self._cache = cache

    def _get_score(self, text, labels):
        key = "---".join(
            [
                text,
                str(self.max_l_dist),
                str(self.near_match_norm),
                str(self.near_match_treshold),
                str(self.near_match_max_length),
            ]
        )

        result = []
        for label in tqdm_v(labels, self.verbose):
            label_key = key + "---" + str(label)
            score = self._cache.get(label_key)
            if score is None:
                score = self._get_one_score(text, label)
                self._cache[label_key] = score
            result.append(score)
        return result
