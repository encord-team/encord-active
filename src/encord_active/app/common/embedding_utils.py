import os
import pickle
from typing import List, Optional, Tuple

import faiss
import numpy as np
import streamlit as st
from faiss import IndexFlatL2

import encord_active.app.common.state as state
from encord_active.lib.common.utils import (
    fix_duplicate_image_orders_in_knn_graph_single_row,
)


@st.experimental_memo(show_spinner=False)
def get_collections(embedding_name: str) -> list[dict]:
    embedding_path = st.session_state.embeddings_dir / embedding_name
    collections = []
    if os.path.isfile(embedding_path):
        with open(embedding_path, "rb") as f:
            collections = pickle.load(f)
    return collections


@st.experimental_memo
def get_collections_and_metadata(embedding_name: str) -> Tuple[list[dict], dict]:
    try:
        collections = get_collections(embedding_name)

        embedding_metadata_file_name = "embedding_classifications_metadata.pkl"
        embedding_metadata_path = st.session_state.embeddings_dir / embedding_metadata_file_name
        if os.path.isfile(embedding_metadata_path):
            with open(embedding_metadata_path, "rb") as f:
                question_hash_to_collection_indexes_local = pickle.load(f)
        else:
            question_hash_to_collection_indexes_local = {}

        return collections, question_hash_to_collection_indexes_local
    except Exception as e:
        return [], {}


def get_key_from_index(collection: dict, question_hash: Optional[str] = None, has_annotation=True) -> str:
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


def get_identifier_to_neighbors(
    collections: list[dict], nearest_indexes: np.ndarray, has_annotation=True
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


@st.experimental_memo
def get_faiss_index_image(_collections: list) -> dict[str, IndexFlatL2]:
    indexes = {}

    for question_hash in st.session_state[state.QUESTION_HASH_TO_COLLECTION_INDEXES]:
        selected_collections = [
            _collections[i] for i in st.session_state[state.QUESTION_HASH_TO_COLLECTION_INDEXES][question_hash]
        ]

        if len(selected_collections) > 10:
            embedding_database = np.stack(list(map(lambda x: x["embedding"], selected_collections)))

            index = faiss.IndexFlatL2(embedding_database.shape[1])
            index.add(embedding_database)  # pylint: disable=no-value-for-parameter

            indexes[question_hash] = index

    return indexes


@st.experimental_memo
def get_faiss_index_object(_collections: list[dict], faiss_index_name: str) -> IndexFlatL2:
    """

    Args:
        _collections: Underscore is used to skip hashing for this object to make this function faster.
        faiss_index_name: Since we skip hashing for collections, we need another parameter for memoization.

    Returns: Faiss Index object for searching embeddings.

    """
    embeddings_list: List[list] = [x["embedding"] for x in _collections]
    embeddings = np.array(embeddings_list).astype(np.float32)

    if len(embeddings.shape) != 2:
        return

    db_index = faiss.IndexFlatL2(embeddings.shape[1])
    db_index.add(embeddings)  # pylint: disable=no-value-for-parameter
    return db_index


@st.experimental_memo
def get_object_keys_having_similarities(_collections: list[dict]) -> dict:
    return {get_key_from_index(collection): i for i, collection in enumerate(_collections)}


@st.experimental_memo
def get_image_keys_having_similarities(_collections: list[dict]) -> dict:
    return {get_key_from_index(collection, has_annotation=False): i for i, collection in enumerate(_collections)}


def add_labeled_image_neighbors_to_cache(image_identifier: str, question_feature_hash: str) -> None:
    collection_id = st.session_state[state.IMAGE_KEYS_HAVING_SIMILARITIES]["_".join(image_identifier.split("_")[:3])]
    collection_item_index = st.session_state[state.QUESTION_HASH_TO_COLLECTION_INDEXES][question_feature_hash].index(
        collection_id
    )
    embedding = np.array([st.session_state[state.COLLECTIONS_IMAGES][collection_id]["embedding"]]).astype(np.float32)
    nearest_distances, nearest_indexes = st.session_state[state.FAISS_INDEX_IMAGE][question_feature_hash].search(
        embedding, int(st.session_state[state.K_NEAREST_NUM] + 1)
    )
    nearest_indexes = fix_duplicate_image_orders_in_knn_graph_single_row(collection_item_index, nearest_indexes)

    temp_list = []
    for nearest_index in nearest_indexes[0, 1:]:
        collection_index = st.session_state[state.QUESTION_HASH_TO_COLLECTION_INDEXES][question_feature_hash][
            nearest_index
        ]
        temp_list.append(
            {
                "key": get_key_from_index(
                    st.session_state[state.COLLECTIONS_IMAGES][collection_index],
                    question_hash=question_feature_hash,
                    has_annotation=st.session_state[state.CURRENT_INDEX_HAS_ANNOTATION],
                ),
                "name": st.session_state[state.COLLECTIONS_IMAGES][collection_index]["classification_answers"][
                    question_feature_hash
                ]["answer_name"],
            }
        )

    st.session_state[state.IMAGE_SIMILARITIES][question_feature_hash][image_identifier] = temp_list


def _get_nearest_items_from_nearest_indexes(nearest_indexes: np.ndarray, collection_type: str) -> list[dict]:
    temp_list = []
    for nearest_index in nearest_indexes[0, 1:]:
        temp_list.append(
            {
                "key": get_key_from_index(
                    st.session_state[collection_type][nearest_index],
                    has_annotation=st.session_state[state.CURRENT_INDEX_HAS_ANNOTATION],
                ),
                "name": st.session_state[collection_type][nearest_index].get("name", "No label"),
            }
        )

    return temp_list


def add_object_neighbors_to_cache(object_identifier: str) -> None:
    item_index = st.session_state[state.OBJECT_KEYS_HAVING_SIMILARITIES][object_identifier]
    embedding = np.array([st.session_state[state.COLLECTIONS_OBJECTS][item_index]["embedding"]]).astype(np.float32)
    nearest_distances, nearest_indexes = st.session_state[state.FAISS_INDEX_OBJECT].search(
        embedding, int(st.session_state[state.K_NEAREST_NUM] + 1)
    )

    st.session_state[state.OBJECT_SIMILARITIES][object_identifier] = _get_nearest_items_from_nearest_indexes(
        nearest_indexes, state.COLLECTIONS_OBJECTS
    )


def add_image_neighbors_to_cache(image_identifier: str):
    collection_id = st.session_state[state.IMAGE_KEYS_HAVING_SIMILARITIES][image_identifier]
    embedding = np.array([st.session_state[state.COLLECTIONS_IMAGES][collection_id]["embedding"]]).astype(np.float32)
    nearest_distances, nearest_indexes = st.session_state[state.FAISS_INDEX_IMAGE_NO_LABEL].search(
        embedding, int(st.session_state[state.K_NEAREST_NUM] + 1)
    )

    st.session_state[state.IMAGE_SIMILARITIES_NO_LABEL][image_identifier] = _get_nearest_items_from_nearest_indexes(
        nearest_indexes, state.COLLECTIONS_IMAGES
    )
