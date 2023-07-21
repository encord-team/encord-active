from abc import ABCMeta, abstractmethod
from typing import Optional, Set
from encord_active.analysis.base import BaseEvaluation, BaseFrameInput, BaseFrameOutput
from encord_active.analysis.types import (
    EmbeddingTensor,
    ImageTensor,
    MaskTensor,
    PointTensor,
)


class BaseEmbedding(BaseEvaluation, metaclass=ABCMeta):
    pass


class PureImageEmbedding(BaseEmbedding, metaclass=ABCMeta):
    """
    Pure image based embedding, can optionally be calculated on a per-object basis as well as per-image by default.
    """

    def __init__(
        self, ident: str, dependencies: Set[str], allow_object_embedding: bool = True, allow_queries: bool = False
    ) -> None:
        super().__init__(ident, dependencies)
        self.allow_object_embedding = allow_object_embedding
        self.allow_queries = allow_queries

    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        annotations = {}
        image = self.evaluate_embedding(frame.image, mask=None)
        if self.allow_object_embedding:
            for annotation_hash, annotation in frame.annotations.items():
                if annotation.mask is None:
                    annotations[annotation_hash] = image
                else:
                    annotations[annotation_hash] = self.evaluate_embedding(
                        image=frame.image,
                        mask=annotation.mask,
                    )
        return BaseFrameOutput(
            image=image,
            annotations=annotations,
        )

    @abstractmethod
    def evaluate_embedding(self, image: ImageTensor, mask: Optional[MaskTensor]) -> EmbeddingTensor:
        ...


class PureObjectEmbedding(BaseEmbedding, metaclass=ABCMeta):
    def __init__(self, ident: str, dependencies: Set[str], allow_queries: bool = False) -> None:
        super().__init__(ident, dependencies)
        self.allow_queries = allow_queries

    """
    Pure object based embedding, only depends on object metadata
    """

    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        # FIXME: implement
        return BaseFrameOutput(image=None, annotations={})

    @abstractmethod
    def evaluate_embedding(self, mask: MaskTensor, points: PointTensor) -> EmbeddingTensor:
        ...


class NearestImageEmbeddingQuery:
    """
    Pseudo embedding, returns the embedding for the nearest image embedding (in the same category).
    Extra dependency tracking is automatically added whenever this is used.
    """

    def __init__(self, ident: str, embedding_source: str, max_neighbors: int = 11) -> None:
        self.ident = ident
        # NOTE: as this-is implemented externally by the executor.
        # dependencies is not set. This returns a reference for up to 'max_neighbours' nearby
        # values and all associated stage 1 metric results. (ONLY stage 1 metric results!)
        self.embedding_source = embedding_source
        self.max_neighbours = max_neighbors

    """
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
    """
