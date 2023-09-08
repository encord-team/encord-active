import uuid
from typing import List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel
from pynndescent import NNDescent

from encord_active.server.dependencies import DataOrAnnotateItem


class SimilarityResult(BaseModel):
    item: str
    similarity: float


def pack_similarity_result(k: Tuple[uuid.UUID, int, Optional[str]], s: float) -> SimilarityResult:
    du_hash, frame, *annotation_hash_opt = k
    annotation_hash = annotation_hash_opt[0] if len(annotation_hash_opt) > 0 else None
    item = DataOrAnnotateItem(du_hash=du_hash, frame=frame, annotation_hash=annotation_hash).pack()
    return SimilarityResult(
        item=item,
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
        return [pack_similarity_result(self.results[idx], similarity) for idx, similarity in zip(indices, distances)]