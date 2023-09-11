from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

import torch

from encord_active.analysis.base import (
    BaseEvaluation,
    BaseFrameBatchInput,
    BaseFrameBatchOutput,
    BaseFrameInput,
    BaseFrameOutput,
)
from encord_active.analysis.metric import ObjectOnlyBatchInput
from encord_active.analysis.types import (
    EmbeddingBatchTensor,
    EmbeddingDistanceMetric,
    EmbeddingTensor,
    ImageBatchTensor,
    ImageTensor,
    MaskTensor,
    MetricResult,
)


class BaseEmbedding(BaseEvaluation, metaclass=ABCMeta):
    pass


@dataclass
class ImageEmbeddingResult:
    images: EmbeddingBatchTensor
    objects: Optional[EmbeddingBatchTensor]


class PureImageEmbedding(BaseEmbedding, metaclass=ABCMeta):
    """
    Pure image based embedding, can optionally be calculated on a per-object basis as well as per-image by default.
    """

    def __init__(self, ident: str, allow_object_embedding: bool = True, allow_queries: bool = False) -> None:
        super().__init__(ident)
        self.allow_object_embedding = allow_object_embedding
        self.allow_queries = allow_queries

    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        annotations: Dict[str, MetricResult] = {}
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

    def raw_calculate_batch(
        self,
        prev_frame: Optional[BaseFrameBatchInput],
        frame: BaseFrameBatchInput,
        next_frame: Optional[BaseFrameBatchInput],
    ) -> BaseFrameBatchOutput:
        res = self.evaluate_embedding_batched(
            frame.images,
            None if frame.annotations is None else frame.annotations,
        )
        classifications = None
        if frame.annotations is not None:
            classifications = torch.index_select(res.images, 0, frame.annotations.classifications_image_indices)
        return BaseFrameBatchOutput(
            images=res.images,
            objects=res.objects,
            classifications=classifications,
        )

    @abstractmethod
    def evaluate_embedding_batched(
        self, image: ImageBatchTensor, objects: Optional[ObjectOnlyBatchInput]
    ) -> ImageEmbeddingResult:
        ...


class PureObjectEmbedding(BaseEmbedding, metaclass=ABCMeta):
    def __init__(self, ident: str, allow_queries: bool = False) -> None:
        super().__init__(ident)
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
        annotations: Dict[str, MetricResult] = {}
        for annotation_hash, annotation in frame.annotations.items():
            if annotation.mask is not None:
                annotations[annotation_hash] = self.evaluate_embedding(annotation.mask)
            else:
                annotations[annotation_hash] = self.classification_embedding()
        return BaseFrameOutput(image=None, annotations=annotations)

    @abstractmethod
    def evaluate_embedding(self, mask: MaskTensor) -> EmbeddingTensor:
        ...

    @abstractmethod
    def classification_embedding(self) -> Optional[EmbeddingTensor]:
        ...

    def raw_calculate_batch(
        self,
        prev_frame: Optional[BaseFrameBatchInput],
        frame: BaseFrameBatchInput,
        next_frame: Optional[BaseFrameBatchInput],
    ) -> BaseFrameBatchOutput:
        raise ValueError()


class NearestImageEmbeddingQuery:
    """
    Pseudo embedding, returns the embedding for the nearest image embedding (in the same category).
    Extra dependency tracking is automatically added whenever this is used.
    """

    def __init__(
        self,
        ident: str,
        embedding_source: str,
        max_neighbors: int = 11,
        similarity: EmbeddingDistanceMetric = EmbeddingDistanceMetric.EUCLIDEAN,
        same_feature_hash: bool = False,
    ) -> None:
        self.ident = ident
        # NOTE: as this-is implemented externally by the executor.
        # This returns a reference for up to 'max_neighbours' nearby
        # values and all associated stage 1 metric results. (ONLY stage 1 metric results!)
        self.embedding_source = embedding_source
        self.max_neighbours = max_neighbors
        self.similarity = similarity
        self.same_feature_hash = same_feature_hash

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
