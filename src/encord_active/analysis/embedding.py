from abc import ABCMeta, abstractmethod
from typing import cast

import torch
from pynndescent import NNDescent
from torch import BoolTensor, FloatTensor, IntTensor

from encord_active.analysis.base import BaseAnalysis, BaseEvaluation
from encord_active.analysis.metric import MetricDependencies
from encord_active.analysis.types import (
    EmbeddingTensor,
    ImageTensor,
    MaskTensor,
    MetricKey,
    MetricResult,
    NearestNeighbors,
    PointTensor,
)


class PureImageEmbedding(BaseEvaluation, metaclass=ABCMeta):
    """
    Pure image based embedding, can optionally be calculated on a per-object basis as well as per-image by default.
    """

    def __init__(
        self, ident: str, dependencies: set[str], allow_object_embedding: bool = True, allow_queries: bool = False
    ) -> None:
        super().__init__(ident, dependencies)
        self.allow_object_embedding = allow_object_embedding
        self.allow_queries = allow_queries

    @abstractmethod
    def evaluate_embedding(self, image: ImageTensor, mask: MaskTensor | None) -> EmbeddingTensor:
        ...

    def evaluate_object_embedding(
        self, image: ImageTensor, mask: MaskTensor, coordinates: PointTensor
    ) -> EmbeddingTensor:
        _coordinates = coordinates  # Only used in override logic for special case shortcuts
        return self.evaluate_embedding(image, mask)


class PureObjectEmbedding(BaseEvaluation, metaclass=ABCMeta):
    def __init__(self, ident: str, dependencies: set[str], allow_queries: bool = False) -> None:
        super().__init__(ident, dependencies)
        self.allow_queries = allow_queries

    """
    Pure object based embedding, only depends on object metadata
    """

    @abstractmethod
    def evaluate_embedding(self, mask: MaskTensor, coordinates: PointTensor) -> EmbeddingTensor:
        ...


class NearestImageEmbeddingQuery(BaseEvaluation):
    """
    Pseudo embedding, returns the embedding for the nearest image embedding (in the same category).
    Extra dependency tracking is automatically added whenever this is used.
    """

    def __init__(self, ident: str, embedding_source: str, max_neighbors: int = 11) -> None:
        super().__init__(ident=ident, dependencies={embedding_source, "feature-hash"})
        self.embedding_source = embedding_source
        self.feature_hashes = []
        self.keys: list[MetricKey] = []
        self.index: NNDescent | None = None

    def setup_embedding_index(self, metric_results: dict[MetricKey, dict[str, MetricResult | EmbeddingTensor]]):
        # TODO split on images and objects
        self.keys = list(metric_results.keys())
        self.feature_hashes = [v["feature-hash"] for v in metric_results.values()]

        embeddings = torch.stack([v[self.embedding_source] for v in metric_results.values()]).detach().cpu().numpy()  # type: ignore
        self.index = NNDescent(embeddings, n_neighbors=50, metric="cosine")
        self.index.prepare()

    def execute(self, metric_result: MetricDependencies) -> NearestNeighbors:
        if not self.index:
            raise RuntimeError(
                "`NearestImageEmbeddingQuery.execute` was called without calling `steup_embedding_index` first"
            )

        embedding = cast(EmbeddingTensor, metric_result[self.embedding_source]).cpu().numpy().unsqueeze(0)
        neighbor_indices, distances = self.index.query(embedding)
        neighbor_indices = neighbor_indices.reshape[-1]
        distances = distances.reshape(-1)
        keys = [self.keys[k].annotation for k in neighbor_indices]
        return NearestNeighbors(keys[1:], distances[1:])
