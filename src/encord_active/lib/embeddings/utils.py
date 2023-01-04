import os
import pickle
from pathlib import Path
from typing import List, Optional, Tuple, TypedDict

import faiss
import numpy as np
from faiss import IndexFlatL2


class LabelEmbedding(TypedDict):
    label_row: str
    data_unit: str
    frame: int
    objectHash: Optional[str]
    lastEditedBy: str
    featureHash: str
    name: str
    dataset_title: str
    embedding: np.ndarray


def get_collections(embedding_name: str, embeddings_dir: Path) -> list[LabelEmbedding]:
    embedding_path = embeddings_dir / embedding_name
    collections = []
    if os.path.isfile(embedding_path):
        with open(embedding_path, "rb") as f:
            collections = pickle.load(f)
    return collections


def get_collections_and_metadata(embedding_name: str, embeddings_dir: Path) -> Tuple[list[LabelEmbedding], dict]:
    try:
        collections = get_collections(embedding_name, embeddings_dir)

        embedding_metadata_file_name = "embedding_classifications_metadata.pkl"
        embedding_metadata_path = embeddings_dir / embedding_metadata_file_name
        if os.path.isfile(embedding_metadata_path):
            with open(embedding_metadata_path, "rb") as f:
                question_hash_to_collection_indexes_local = pickle.load(f)
        else:
            question_hash_to_collection_indexes_local = {}

        return collections, question_hash_to_collection_indexes_local
    except Exception:
        return [], {}


def get_key_from_index(collection: LabelEmbedding, question_hash: Optional[str] = None, has_annotation=True) -> str:
    label_hash = collection["label_row"]
    du_hash = collection["data_unit"]
    frame_idx = int(collection["frame"])

    if not has_annotation:
        key = f"{label_hash}_{du_hash}_{frame_idx:05d}"
    else:
        if question_hash:
            key = f"{label_hash}_{du_hash}_{frame_idx:05d}_{question_hash}"
        else:
            object_hash = collection["objectHash"]
            key = f"{label_hash}_{du_hash}_{frame_idx:05d}_{object_hash}"

    return key


# TODO: remove if unused
def get_identifier_to_neighbors(
    collections: list[LabelEmbedding], nearest_indexes: np.ndarray, has_annotation=True
) -> dict[str, list]:
    nearest_neighbors = {}
    n, k = nearest_indexes.shape
    for i in range(n):
        key = get_key_from_index(collections[i], has_annotation=has_annotation)
        temp_list = []
        for j in range(1, k):
            temp_list.append(
                {
                    "key": get_key_from_index(collections[nearest_indexes[i, j]], has_annotation=has_annotation),
                    "name": collections[nearest_indexes[i, j]].get("name", "Does not have a label"),
                }
            )
        nearest_neighbors[key] = temp_list
    return nearest_neighbors


# TODO: remove if unused
def convert_to_indexes(collections, question_hash_to_collection_indexes):
    embedding_databases, indexes = {}, {}

    for question_hash in question_hash_to_collection_indexes:
        selected_collections = [collections[i] for i in question_hash_to_collection_indexes[question_hash]]

        if len(selected_collections) > 10:
            embedding_database = np.stack(list(map(lambda x: x["embedding"], selected_collections)))

            index = faiss.IndexFlatL2(embedding_database.shape[1])
            index.add(embedding_database)  # pylint: disable=no-value-for-parameter

            embedding_databases[question_hash] = embedding_database
            indexes[question_hash] = index

    return embedding_databases, indexes


def get_faiss_index_image(
    collections: list[LabelEmbedding], question_hash_to_collection_indexes: dict
) -> dict[str, IndexFlatL2]:
    indexes = {}

    for question_hash in question_hash_to_collection_indexes:
        selected_collections = [collections[i] for i in question_hash_to_collection_indexes[question_hash]]

        if len(selected_collections) > 10:
            embedding_database = np.stack(list(map(lambda x: x["embedding"], selected_collections)))

            index = faiss.IndexFlatL2(embedding_database.shape[1])
            index.add(embedding_database)  # pylint: disable=no-value-for-parameter

            indexes[question_hash] = index

    return indexes


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


def get_object_keys_having_similarities(collections: list[LabelEmbedding]) -> dict:
    return {get_key_from_index(collection): i for i, collection in enumerate(collections)}


def get_image_keys_having_similarities(collections: list[LabelEmbedding]) -> dict:
    return {get_key_from_index(collection, has_annotation=False): i for i, collection in enumerate(collections)}
