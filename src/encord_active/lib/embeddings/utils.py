import os
import pickle
from pathlib import Path
from typing import Dict, List, Optional, TypedDict

import faiss
import numpy as np
import pandera as pa
from faiss import IndexFlatL2
from pandera.typing import Series

from encord_active.lib.metrics.metric import EmbeddingType
from encord_active.lib.metrics.utils import IdentifierSchema


class ClassificationAnswer(TypedDict):
    answer_featureHash: str
    answer_name: str
    annotator: str


class LabelEmbedding(TypedDict):
    label_row: str
    data_unit: str
    frame: int
    url: str
    labelHash: Optional[str]
    lastEditedBy: Optional[str]
    featureHash: Optional[str]
    name: Optional[str]
    dataset_title: str
    embedding: np.ndarray
    classification_answers: Optional[ClassificationAnswer]


class SimilaritiesFinder:
    def __init__(self, type: EmbeddingType, path: Path, num_of_neighbors: int = 8):
        self.type = type
        self.path = path
        self.num_of_neighbors = num_of_neighbors
        self.collections = load_collections(self.type, self.path)
        self.faiss_index = get_faiss_index_object(self.collections)
        self.keys_having_similarity = build_keys_having_similarities(self.collections, self.has_annotations)
        self.similarities: Dict[str, List[Dict]] = {}

    @property
    def has_annotations(self):
        return self.type in [EmbeddingType.OBJECT, EmbeddingType.CLASSIFICATION]

    def get_similarities(self, identifier: str):
        if identifier not in self.similarities.keys():
            self._add_similarities(identifier)

        return self.similarities[identifier]

    def _add_similarities(self, identifier: str):
        collection_id = self.keys_having_similarity[identifier]
        embedding = np.array([self.collections[collection_id]["embedding"]]).astype(np.float32)
        _, nearest_indexes = self.faiss_index.search(  # pylint: disable=no-value-for-parameter
            embedding, self.num_of_neighbors + 1
        )

        self.similarities[identifier] = []

        for nearest_index in nearest_indexes[0, 1:]:
            collection = self.collections[nearest_index]

            answers = collection.get("classification_answers") or {}
            if self.type == EmbeddingType.CLASSIFICATION and not answers:
                raise Exception("Missing classification answers")

            key = get_key_from_index(collection, has_annotation=self.has_annotations)
            name = answers.get("answer_name") or collection.get("name", "No label")
            self.similarities[identifier].append({"key": key, "name": name})


EMBEDDING_TYPE_TO_FILENAME = {
    EmbeddingType.IMAGE: "cnn_images.pkl",
    EmbeddingType.CLASSIFICATION: "cnn_classifications.pkl",
    EmbeddingType.OBJECT: "cnn_objects.pkl",
}

EMBEDDING_REDUCED_TO_FILENAME = {
    EmbeddingType.IMAGE: "cnn_images_reduced.pkl",
    EmbeddingType.CLASSIFICATION: "cnn_classifications_reduced.pkl",
    EmbeddingType.OBJECT: "cnn_objects_reduced.pkl",
}


class Embedding2DSchema(IdentifierSchema):
    x: Series[float] = pa.Field(coerce=True)
    y: Series[float] = pa.Field(coerce=True)
    label: Series[str]
    x_y: Series[str]


def load_collections(embedding_type: EmbeddingType, embeddings_dir: Path) -> list[LabelEmbedding]:
    embedding_path = embeddings_dir / EMBEDDING_TYPE_TO_FILENAME[embedding_type]
    collections = []
    if os.path.isfile(embedding_path):
        with open(embedding_path, "rb") as f:
            collections = pickle.load(f)
    return collections


def save_collections(embedding_type: EmbeddingType, embeddings_dir: Path, collection: list[LabelEmbedding]):
    embedding_path = embeddings_dir / EMBEDDING_TYPE_TO_FILENAME[embedding_type]
    with open(embedding_path, "wb") as f:
        pickle.dump(collection, f)


def build_keys_having_similarities(collections: List[LabelEmbedding], has_annotation: bool):
    return {get_key_from_index(collection, has_annotation): i for i, collection in enumerate(collections)}


def get_key_from_index(collection: LabelEmbedding, has_annotation: bool) -> str:
    label_hash = collection["label_row"]
    du_hash = collection["data_unit"]
    frame_idx = int(collection["frame"])

    key = f"{label_hash}_{du_hash}_{frame_idx:05d}"

    if has_annotation:
        key += f"_{collection['labelHash']}"

    return key


def get_faiss_index_object(collections: list[LabelEmbedding]) -> IndexFlatL2:
    """

    Args:
        _collections: Underscore is used to skip hashing for this object to make this function faster.
        faiss_index_name: Since we skip hashing for collections, we need another parameter for memoization.

    Returns: Faiss Index object for searching embeddings.

    """
    embeddings_list: List[np.ndarray] = [x["embedding"] for x in collections]
    embeddings = np.array(embeddings_list).astype(np.float32)

    if len(embeddings.shape) != 2:
        return

    db_index = faiss.IndexFlatL2(embeddings.shape[1])
    db_index.add(embeddings)  # pylint: disable=no-value-for-parameter
    return db_index
