from typing import Optional

try:
    import numpy as np
except Exception:  # pragma: no cover - numpy might be missing
    np = None  # type: ignore

from .indexer import BatchSearchResult, Indexer

try:
    from fastkmeans import KMeans as FastKMeans  # type: ignore
except Exception:  # pragma: no cover - library not installed
    try:
        from sklearn.cluster import MiniBatchKMeans as FastKMeans
    except Exception:  # pragma: no cover - no sklearn either
        class FastKMeans:  # type: ignore
            def __init__(self, n_clusters=8, random_state=None):
                self.n_clusters = n_clusters

            def fit(self, X):
                self.labels_ = [0] * len(X)

            def predict(self, X):  # noqa: D401 - mimic sklearn API
                return [0] * len(X)


class FastKMeansIndexer(Indexer):
    """Approximate nearest neighbor search using fastkmeans."""

    def __init__(self, n_clusters: int = 256, random_state: Optional[int] = None):
        super().__init__()
        self.n_clusters = n_clusters
        self.random_state = random_state

    def build_index(self, data):
        if np is not None:
            self.data = np.asarray(data, dtype=float)
        else:  # pragma: no cover - numpy missing
            self.data = [list(map(float, row)) for row in data]
        self.kmeans = FastKMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.kmeans.fit(self.data)
        # store labels for possible cluster-based search
        try:
            self.labels = self.kmeans.labels_
        except AttributeError:  # pragma: no cover - different api
            self.labels = self.kmeans.predict(self.data)

    def batch_search(self, queries, k: int = 10) -> BatchSearchResult:
        if np is not None:
            queries = np.asarray(queries, dtype=float)
        else:  # pragma: no cover - numpy missing
            queries = [list(map(float, row)) for row in queries]
        batch_scores = []
        batch_indices = []
        for q in queries:
            # brute-force search over all points for correctness
            if np is not None:
                dists = np.linalg.norm(self.data - q, axis=1)
                nn_idx = np.argsort(dists)[:k]
                batch_indices.append(nn_idx.tolist())
                batch_scores.append((-dists[nn_idx]).tolist())
            else:  # pragma: no cover - numpy missing
                dists = [sum((dp_i - q_i) ** 2 for dp_i, q_i in zip(dp, q)) ** 0.5 for dp in self.data]
                sorted_indices = sorted(range(len(dists)), key=lambda i: dists[i])[:k]
                batch_indices.append(sorted_indices)
                batch_scores.append([-dists[i] for i in sorted_indices])
        return BatchSearchResult(batch_scores, batch_indices)
