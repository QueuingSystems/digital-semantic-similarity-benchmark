from sklearn.base import BaseEstimator, ClassifierMixin

__all__ = ["KeywordsClusterNew"]


class KeywordsClusterNew(BaseEstimator, ClassifierMixin):
    def __init__(self, cluster_model, source_selection=None):
        self.cluster_model = cluster_model
        self.source_selection = source_selection

    def filter_keywords(self, kws):
        result = []
        for kw in kws:
            if len(kw["value"]) < 85:
                result.append(kw)
        return result

    def select_best(self, kws):
        if self.source_selection:
            sources = set([kw["source"] for kw in kws])
            for source in self.source_selection:
                if source in sources:
                    kws = [kw for kw in kws if kw["source"] == source]
                    break
        max_score = max([kw["certainty"] for kw in kws])
        top_33 = [kw for kw in kws if kw["certainty"] > max_score * 0.66]
        top_33 = sorted(top_33, key=lambda kw: len(kw["value"]), reverse=True)
        return top_33[0]

    def filter_subsets_(self, kws):
        values = []
        result = []
        for kw in kws:
            contained = False
            for value in values:
                if kw["value"] in value:
                    contained = True
                    break
            if contained:
                continue

            result.append(kw)
            values.append(kw["value"])
        return result

    def filter_subsets(self, kws):
        distinct_values = {}
        for kw in kws:
            contained = False
            for i, value in distinct_values.items():
                if kw["value"] in value:
                    contained = True
                    break
                if value in kw["value"]:
                    distinct_values[i] = kw["value"]
                    contained = True
            if contained:
                continue
            distinct_values[len(distinct_values)] = kw["value"]
        distinct = set(distinct_values.values())
        result = []
        for kw in kws:
            if kw["value"] in distinct:
                result.append(kw)
        return result

    def _cluster_keywords(self, datum):
        datum = self.filter_keywords(datum)
        clusters = self.cluster_model.fit_predict([kw["value"] for kw in datum])
        kw_by_cluster = {}
        for cluster, kw in zip(clusters, datum):
            try:
                kw_by_cluster[cluster].append(kw)
            except KeyError:
                kw_by_cluster[cluster] = [kw]
        result = []
        for cluster, kws in kw_by_cluster.items():
            kws = sorted(kws, key=lambda v: v["certainty"], reverse=True)
            # value = kws[0]['value']
            value = self.select_best(kws)["value"]
            score = min(sum([kw["certainty"] for kw in kws]), 1)
            result.append(
                {
                    "value": value,
                    "certainty": score,
                    "meta": {
                        "items": [{"v": kw["value"], "s": kw["source"]} for kw in kws]
                    },
                }
            )
        result = sorted(result, key=lambda v: v["certainty"], reverse=True)
        result = self.filter_subsets(result)
        return result

    def fit(self, X=None, Y=None):
        pass

    def transform(self, X):
        return self.predict(X)

    def predict(self, X):
        for x in X:
            yield self._cluster_keywords(x)
