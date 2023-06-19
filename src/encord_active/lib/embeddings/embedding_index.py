from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
from pynndescent import NNDescent

from encord_active.lib.common.iterator import DatasetIterator, Iterator
from encord_active.lib.embeddings.embeddings import get_embeddings
from encord_active.lib.embeddings.types import LabelEmbedding
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.project.project_file_structure import ProjectFileStructure

MAX_BATCH_SIZE = int(os.getenv("EA_NN_SEARCH_MAX_QUERY_SIZE", 20_000))


class EmbeddingSearchResult(NamedTuple):
    indices: np.ndarray
    # [m, k] array
    similarities: np.ndarray
    # [m, k] array


def _query_in_batches(index: NNDescent, embeddings: np.ndarray, k: int, batch_size=1000) -> EmbeddingSearchResult:
    n = embeddings.shape[0]
    if n <= batch_size:
        return EmbeddingSearchResult(*index.query(embeddings, k=k))

    dists, idxs = [], []
    for idx in range(0, n, batch_size):
        d, i = index.query(embeddings[idx : idx + batch_size], k=k)
        dists.append(d)
        idxs.append(i)
    return EmbeddingSearchResult(np.concatenate(dists, axis=0), np.concatenate(idxs, axis=0))


def _get_embedding_index_file(embedding_file: Path, metric: str) -> Path:
    return embedding_file.parent / f"{embedding_file.stem}_{metric}_index.pkl"


class EmbeddingIndex:
    @classmethod
    def index_available(
        cls, project_file_structure: ProjectFileStructure, embedding_type: EmbeddingType, metric: str = "cosine"
    ):
        embedding_file = project_file_structure.get_embeddings_file(embedding_type)
        return _get_embedding_index_file(embedding_file, metric).is_file()

    @classmethod
    def from_project(
        cls,
        project_file_structure: ProjectFileStructure,
        embedding_type: EmbeddingType,
        iterator: Optional[Iterator] = None,
        metric: str = "cosine",
    ) -> tuple[Optional[EmbeddingIndex], list[LabelEmbedding]]:
        """
        Get a stored embedding index from disc. If it doesn't exist, it'll be computed on the fly.

        Args:
            project_file_structure: The project
            embedding_type: The embedding type for which the metric should apply.
            iterator: An optional iterator to avoid instantiating it again. If none, a new one will be instantiated.
            metric: Multiple metrics are supported. For more, see [here](https://pynndescent.readthedocs.io/en/latest/pynndescent_metrics.html#Built-in-metrics)

        Returns:
            An index ready for querying

        """
        embedding_file = project_file_structure.get_embeddings_file(embedding_type)
        index_file = _get_embedding_index_file(embedding_file, metric)

        if iterator is None:
            iterator = DatasetIterator(cache_dir=project_file_structure.project_dir)
        embeddings = get_embeddings(iterator, embedding_type)

        if embeddings == []:
            return None, []

        idx: EmbeddingIndex
        try:
            assert index_file.is_file()
            idx = pickle.loads(index_file.read_bytes())
        except:
            np_embeddings = np.stack([e["embedding"] for e in embeddings]).astype(np.float32)
            idx = EmbeddingIndex(np_embeddings, metric=metric)
            idx.prepare()
            index_file.write_bytes(pickle.dumps(idx))
        return idx, embeddings

    def __init__(self, embeddings: np.ndarray, inplace=True, metric: str = "cosine") -> None:
        self.n, self.d = embeddings.shape
        self.inplace = inplace

        self.index = NNDescent(embeddings, n_neighbors=50, metric=metric)

    def prepare(self):
        self.index.prepare()

    def query(self, query_embeddings: np.ndarray, k: Optional[int] = None):
        """
        The query function.

        Args:
            query_embeddings: Either an [m, d] array of m queries or a [d] array with one query
            k: Number of neighbors to fetch. If k is None, order all items in the index.

        Returns:
            The search results

        """
        if k is None:
            k = self.n
        else:
            k = min(self.n, k)

        _query_embeddings = query_embeddings
        if _query_embeddings.ndim == 1:
            _query_embeddings = _query_embeddings.reshape(1, -1)

        if _query_embeddings.shape[0] > MAX_BATCH_SIZE:
            return _query_in_batches(self.index, _query_embeddings, k=k)

        return EmbeddingSearchResult(*self.index.query(_query_embeddings, k=k))
