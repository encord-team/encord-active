import numpy as np
import streamlit as st
from pandas import Series
from streamlit.delta_generator import DeltaGenerator

from encord_active.app.common.state import (
    COLLECTIONS_IMAGES,
    COLLECTIONS_OBJECTS,
    CURRENT_INDEX_HAS_ANNOTATION,
    FAISS_INDEX_IMAGE,
    FAISS_INDEX_IMAGE_NO_LABEL,
    FAISS_INDEX_OBJECT,
    IMAGE_KEYS_HAVING_SIMILARITIES,
    IMAGE_SIMILARITIES,
    IMAGE_SIMILARITIES_NO_LABEL,
    K_NEAREST_NUM,
    OBJECT_KEYS_HAVING_SIMILARITIES,
    OBJECT_SIMILARITIES,
    QUESTION_HASH_TO_COLLECTION_INDEXES,
)
from encord_active.app.common.state_new import get_state
from encord_active.lib.common.image_utils import (
    load_or_fill_image,
    show_image_and_draw_polygons,
)
from encord_active.lib.common.utils import (
    fix_duplicate_image_orders_in_knn_graph_single_row,
)
from encord_active.lib.embeddings.utils import get_key_from_index


def show_similar_classification_images(row: Series, expander: DeltaGenerator):
    feature_hash = row["identifier"].split("_")[-1]

    if row["identifier"] not in st.session_state[IMAGE_SIMILARITIES][feature_hash].keys():
        add_labeled_image_neighbors_to_cache(row["identifier"], feature_hash)

    nearest_images = st.session_state[IMAGE_SIMILARITIES][feature_hash][row["identifier"]]

    division = 4
    column_id = 0
    st_columns = []

    for nearest_image in nearest_images:
        if column_id == 0:
            st_columns = expander.columns(division)

        image = load_or_fill_image(nearest_image["key"], global_state.project_paths.data)

        st_columns[column_id].image(image)
        st_columns[column_id].write(f"Annotated as `{nearest_image['name']}`")
        column_id += 1
        column_id = column_id % division


def show_similar_images(row: Series, expander: DeltaGenerator):
    image_identifier = "_".join(row["identifier"].split("_")[:3])

    if image_identifier not in st.session_state[IMAGE_SIMILARITIES_NO_LABEL].keys():
        add_image_neighbors_to_cache(image_identifier)

    nearest_images = st.session_state[IMAGE_SIMILARITIES_NO_LABEL][image_identifier]

    division = 4
    column_id = 0
    st_columns = []

    for nearest_image in nearest_images:
        if column_id == 0:
            st_columns = expander.columns(division)

        image = load_or_fill_image(nearest_image["key"], global_state.project_paths.data)

        st_columns[column_id].image(image)
        st_columns[column_id].write(f"Annotated as `{nearest_image['name']}`")
        column_id += 1
        column_id = column_id % division


def show_similar_object_images(row: Series, expander: DeltaGenerator):
    object_identifier = "_".join(row["identifier"].split("_")[:4])

    if object_identifier not in st.session_state[OBJECT_KEYS_HAVING_SIMILARITIES]:
        expander.write("Similarity search is not available for this object.")
        return

    if object_identifier not in st.session_state[OBJECT_SIMILARITIES].keys():
        add_object_neighbors_to_cache(object_identifier)

    nearest_images = st.session_state[OBJECT_SIMILARITIES][object_identifier]

    division = 4
    column_id = 0
    st_columns = []

    for nearest_image in nearest_images:
        if column_id == 0:
            st_columns = expander.columns(division)

        image = show_image_and_draw_polygons(nearest_image["key"], get_state().project_paths.data)

        st_columns[column_id].image(image)
        st_columns[column_id].write(f"Annotated as `{nearest_image['name']}`")
        column_id += 1
        column_id = column_id % division


def add_labeled_image_neighbors_to_cache(image_identifier: str, question_feature_hash: str) -> None:
    collection_id = st.session_state[IMAGE_KEYS_HAVING_SIMILARITIES]["_".join(image_identifier.split("_")[:3])]
    collection_item_index = st.session_state[QUESTION_HASH_TO_COLLECTION_INDEXES][question_feature_hash].index(
        collection_id
    )
    embedding = np.array([st.session_state[COLLECTIONS_IMAGES][collection_id]["embedding"]]).astype(np.float32)
    _, nearest_indexes = st.session_state[FAISS_INDEX_IMAGE][question_feature_hash].search(
        embedding, int(st.session_state[K_NEAREST_NUM] + 1)
    )
    nearest_indexes = fix_duplicate_image_orders_in_knn_graph_single_row(collection_item_index, nearest_indexes)

    temp_list = []
    for nearest_index in nearest_indexes[0, 1:]:
        collection_index = st.session_state[QUESTION_HASH_TO_COLLECTION_INDEXES][question_feature_hash][nearest_index]
        temp_list.append(
            {
                "key": get_key_from_index(
                    st.session_state[COLLECTIONS_IMAGES][collection_index],
                    question_hash=question_feature_hash,
                    has_annotation=st.session_state[CURRENT_INDEX_HAS_ANNOTATION],
                ),
                "name": st.session_state[COLLECTIONS_IMAGES][collection_index]["classification_answers"][
                    question_feature_hash
                ]["answer_name"],
            }
        )

    st.session_state[IMAGE_SIMILARITIES][question_feature_hash][image_identifier] = temp_list


def _get_nearest_items_from_nearest_indexes(nearest_indexes: np.ndarray, collection_type: str) -> list[dict]:
    temp_list = []
    for nearest_index in nearest_indexes[0, 1:]:
        temp_list.append(
            {
                "key": get_key_from_index(
                    st.session_state[collection_type][nearest_index],
                    has_annotation=st.session_state[CURRENT_INDEX_HAS_ANNOTATION],
                ),
                "name": st.session_state[collection_type][nearest_index].get("name", "No label"),
            }
        )

    return temp_list


def add_object_neighbors_to_cache(object_identifier: str) -> None:
    item_index = st.session_state[OBJECT_KEYS_HAVING_SIMILARITIES][object_identifier]
    embedding = np.array([st.session_state[COLLECTIONS_OBJECTS][item_index]["embedding"]]).astype(np.float32)
    _, nearest_indexes = st.session_state[FAISS_INDEX_OBJECT].search(
        embedding, int(st.session_state[K_NEAREST_NUM] + 1)
    )

    st.session_state[OBJECT_SIMILARITIES][object_identifier] = _get_nearest_items_from_nearest_indexes(
        nearest_indexes, COLLECTIONS_OBJECTS
    )


def add_image_neighbors_to_cache(image_identifier: str):
    collection_id = st.session_state[IMAGE_KEYS_HAVING_SIMILARITIES][image_identifier]
    embedding = np.array([st.session_state[COLLECTIONS_IMAGES][collection_id]["embedding"]]).astype(np.float32)
    _, nearest_indexes = st.session_state[FAISS_INDEX_IMAGE_NO_LABEL].search(
        embedding, int(st.session_state[K_NEAREST_NUM] + 1)
    )

    st.session_state[IMAGE_SIMILARITIES_NO_LABEL][image_identifier] = _get_nearest_items_from_nearest_indexes(
        nearest_indexes, COLLECTIONS_IMAGES
    )
