import os
import pickle
from functools import partial
from typing import Dict, List, Optional

import numpy as np

from encord_active.lib.embeddings.embedding_index import EmbeddingIndex
from encord_active.lib.embeddings.types import LabelEmbedding
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.project.project_file_structure import ProjectFileStructure


class SimilaritiesFinder:
    def __init__(
        self,
        embedding_type: EmbeddingType,
        project_file_structure: ProjectFileStructure,
    ):
        self.embedding_type = embedding_type
        self.similarities: Dict[str, List[str]] = {}

        self.index = self.label_embeddings = self.identifier_to_index = None
        if EmbeddingIndex.index_available(project_file_structure, embedding_type):
            self.index, self.label_embeddings = EmbeddingIndex.from_project(project_file_structure, embedding_type)
            self.identifier_to_index = build_identifier_to_idx_map(self.label_embeddings, self.has_annotations)

    @property
    def index_available(self) -> bool:
        return self.index is not None

    @property
    def has_annotations(self):
        return self.embedding_type in [EmbeddingType.OBJECT, EmbeddingType.CLASSIFICATION]

    def get_similarities(self, identifier: str, num_neighbors: Optional[int] = None) -> list[str]:
        if not self.index_available or self.identifier_to_index is None or identifier not in self.identifier_to_index:
            return []

        if identifier not in self.similarities.keys():
            self._add_similarities(identifier, num_neighbors)

        return self.similarities[identifier]

    def _add_similarities(self, identifier: str, num_neighbors: Optional[int] = None):
        if self.index is None or self.label_embeddings is None or self.identifier_to_index is None:
            return

        embedding_idx = self.identifier_to_index[identifier]
        embedding = np.array([self.label_embeddings[embedding_idx]["embedding"]]).astype(np.float32)
        search_result = self.index.query(embedding, k=min(1001, num_neighbors or self.index.n))
        _get_embedding = partial(get_embedding_identifier, has_annotation=self.has_annotations)
        self.similarities[identifier] = list(
            map(lambda idx: _get_embedding(self.label_embeddings[int(idx)]), search_result.indices[0, 1:])  # type: ignore
        )


def load_label_embeddings(
    embedding_type: EmbeddingType, project_file_structure: ProjectFileStructure
) -> list[LabelEmbedding]:
    embedding_path = project_file_structure.get_embeddings_file(embedding_type)
    label_embeddings = []
    if os.path.isfile(embedding_path):
        with open(embedding_path, "rb") as f:
            label_embeddings = pickle.load(f)
    return label_embeddings


def save_label_embeddings(
    embedding_type: EmbeddingType, project_file_structure: ProjectFileStructure, label_embeddings: list[LabelEmbedding]
):
    embedding_path = project_file_structure.get_embeddings_file(embedding_type)
    with open(embedding_path, "wb") as f:
        pickle.dump(label_embeddings, f)


def build_identifier_to_idx_map(label_embeddings: List[LabelEmbedding], has_annotation: bool):
    return {get_embedding_identifier(le, has_annotation): i for i, le in enumerate(label_embeddings)}


def get_embedding_identifier(label_embedding: LabelEmbedding, has_annotation: bool) -> str:
    label_hash = label_embedding["label_row"]
    du_hash = label_embedding["data_unit"]
    frame_idx = int(label_embedding["frame"])

    key = f"{label_hash}_{du_hash}_{frame_idx:05d}"

    if has_annotation:
        key += f"_{label_embedding['labelHash']}"

    return key
