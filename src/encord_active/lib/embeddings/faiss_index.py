import platform
from typing import NamedTuple, Optional

import faiss
import numpy as np
from faiss import normalize_L2


class IPSearchResult(NamedTuple):
    distances: np.ndarray
    indices: np.ndarray


def _query_one_by_one(index: faiss.IndexFlatIP, embeddings: np.ndarray, k: int) -> IPSearchResult:
    if embeddings.shape[0] == 1:
        return IPSearchResult(*index.search(embeddings, k=k))  # pylint: disable=no-value-for-parameter

    dists, idxs = [], []
    for emb in embeddings:
        d, i = index.search(emb[None], k=k)  # pylint: disable=no-value-for-parameter
        dists.append(d)
        idxs.append(i)
    return IPSearchResult(np.concatenate(dists, axis=0), np.concatenate(idxs, axis=0))


class FaissIndex:
    def __init__(self, embeddings: np.ndarray, normalize=True, inplace=True) -> None:
        self.embeddings = embeddings if inplace else embeddings.copy()
        self.normalize = normalize
        self.inplace = inplace

        if normalize:
            normalize_L2(self.embeddings)

        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)  # pylint: disable=no-value-for-parameter

    def query(self, query_embeddings: np.ndarray, k: Optional[int] = None):
        if k is None:
            k = self.embeddings.shape[0]

        _query_embeddings = query_embeddings if self.inplace else query_embeddings.copy()

        if _query_embeddings.ndim == 1:
            _query_embeddings = query_embeddings[None]

        if self.normalize:
            normalize_L2(_query_embeddings)

        if "x86" in platform.machine():
            return _query_one_by_one(self.index, _query_embeddings, k=k)

        return IPSearchResult(*self.index.search(_query_embeddings, k=k))  # pylint: disable=no-value-for-parameter
