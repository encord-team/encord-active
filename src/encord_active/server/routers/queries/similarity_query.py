import uuid
from typing import List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel
from pynndescent import NNDescent


class SimilarityResult(BaseModel):
    du_hash: uuid.UUID
    frame: int
    annotation_hash: Optional[str]
    similarity: float


def _similarity_result(k: Tuple[uuid.UUID, int, Optional[str]], s: float) -> SimilarityResult:
    du_hash, frame, annotation_hash = k
    return SimilarityResult(
        du_hash=du_hash,
        frame=frame,
        annotation_hash=annotation_hash,
        similarity=s,
    )


class SimilarityQuery:
    query_impl: Union[np.ndarray, NNDescent]
    results: List[Tuple[uuid.UUID, int, Optional[str]]]

    def __init__(self, embeddings: List[np.ndarray], results: List[Tuple[uuid.UUID, int, Optional[str]]]) -> None:
        embeddings_stack: np.ndarray = np.stack(embeddings)
        if len(embeddings) <= 4096:
            self.query_impl = embeddings_stack
        else:
            nn: NNDescent = NNDescent(data=embeddings_stack)
            nn.prepare()
            self.query_impl = nn
        self.results = results

    def query(self, embedding: np.ndarray, k: int) -> List[SimilarityResult]:
        if isinstance(self.query, NNDescent):
            indices_stack, distances_stack = self.query.query(embedding.reshape(1, -1), k=k)
            indices = indices_stack.reshape(-1)
            distances = distances_stack.reshape(-1)
        else:
            offsets = self.query - embedding
            similarities = np.linalg.norm(offsets, axis=1)
            indices = np.argsort(similarities)[:k]
            distances = similarities[indices]
        return [_similarity_result(self.results[idx], similarity) for idx, similarity in zip(indices, distances)]
