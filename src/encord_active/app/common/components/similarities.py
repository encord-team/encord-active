import numpy as np
from pandas import Series
from streamlit.delta_generator import DeltaGenerator

from encord_active.app.common.state import get_state
from encord_active.lib.common.image_utils import (
    load_or_fill_image,
    show_image_and_draw_polygons,
)
from encord_active.lib.common.utils import (
    fix_duplicate_image_orders_in_knn_graph_single_row,
)
from encord_active.lib.embeddings.embedding import EmbeddingInformation
from encord_active.lib.embeddings.utils import get_key_from_index


def show_similar_classification_images(
    row: Series, expander: DeltaGenerator, embedding_information: EmbeddingInformation
):
    feature_hash = row["identifier"].split("_")[-1]

    if row["identifier"] not in embedding_information.similarities[feature_hash].keys():
        add_labeled_image_neighbors_to_cache(row["identifier"], feature_hash, embedding_information)

    nearest_images = embedding_information.similarities[feature_hash][row["identifier"]]

    division = 4
    column_id = 0
    st_columns = []

    for nearest_image in nearest_images:
        if column_id == 0:
            st_columns = expander.columns(division)

        image = load_or_fill_image(nearest_image["key"], get_state().project_paths.data)

        st_columns[column_id].image(image)
        st_columns[column_id].write(f"Annotated as `{nearest_image['name']}`")
        column_id += 1
        column_id = column_id % division


def show_similar_images(row: Series, expander: DeltaGenerator, embedding_information: EmbeddingInformation):
    image_identifier = "_".join(row["identifier"].split("_")[:3])

    if image_identifier not in embedding_information.similarities.keys():
        add_image_neighbors_to_cache(image_identifier, embedding_information)

    nearest_images = embedding_information.similarities[image_identifier]

    division = 4
    column_id = 0
    st_columns = []

    for nearest_image in nearest_images:
        if column_id == 0:
            st_columns = expander.columns(division)

        image = load_or_fill_image(nearest_image["key"], get_state().project_paths.data)

        st_columns[column_id].image(image)
        st_columns[column_id].write(f"Annotated as `{nearest_image['name']}`")
        column_id += 1
        column_id = column_id % division


def show_similar_object_images(row: Series, expander: DeltaGenerator, embedding_information: EmbeddingInformation):
    object_identifier = "_".join(row["identifier"].split("_")[:4])

    if object_identifier not in embedding_information.keys_having_similarity:
        expander.write("Similarity search is not available for this object.")
        return

    if object_identifier not in embedding_information.similarities.keys():
        add_object_neighbors_to_cache(object_identifier, embedding_information)

    nearest_images = embedding_information.similarities[object_identifier]

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


def add_labeled_image_neighbors_to_cache(
    image_identifier: str, question_feature_hash: str, embedding_information: EmbeddingInformation
) -> None:
    collection_id = embedding_information.keys_having_similarity["_".join(image_identifier.split("_")[:3])]
    collection_item_index = embedding_information.question_hash_to_collection_indexes[question_feature_hash].index(
        collection_id
    )
    embedding = np.array([embedding_information.collections[collection_id]["embedding"]]).astype(np.float32)
    _, nearest_indexes = embedding_information.faiss_index_mapping[question_feature_hash].search(
        embedding, get_state().similarities_count + 1
    )
    nearest_indexes = fix_duplicate_image_orders_in_knn_graph_single_row(collection_item_index, nearest_indexes)

    temp_list = []
    for nearest_index in nearest_indexes[0, 1:]:
        collection_index = embedding_information.question_hash_to_collection_indexes[question_feature_hash][
            nearest_index
        ]

        collection = embedding_information.collections[collection_index]
        answers = collection["classification_answers"]
        if not answers:
            raise Exception("Missing classification answers")

        temp_list.append(
            {
                "key": get_key_from_index(
                    collection,
                    question_hash=question_feature_hash,
                    has_annotation=embedding_information.has_annotations,
                ),
                "name": answers[question_feature_hash]["answer_name"],
            }
        )

    embedding_information.similarities[question_feature_hash][image_identifier] = temp_list


def _get_nearest_items_from_nearest_indexes(
    nearest_indexes: np.ndarray, embedding_information: EmbeddingInformation
) -> list[dict]:
    temp_list = []
    for nearest_index in nearest_indexes[0, 1:]:
        temp_list.append(
            {
                "key": get_key_from_index(
                    embedding_information.collections[nearest_index],
                    has_annotation=embedding_information.has_annotations,
                ),
                "name": embedding_information.collections[nearest_index].get("name", "No label"),
            }
        )

    return temp_list


def add_object_neighbors_to_cache(object_identifier: str, embedding_information: EmbeddingInformation) -> None:
    if not embedding_information.faiss_index:
        return

    item_index = embedding_information.keys_having_similarity[object_identifier]
    embedding = np.array([embedding_information.collections[item_index]["embedding"]]).astype(np.float32)
    _, nearest_indexes = embedding_information.faiss_index.search(embedding, get_state().similarities_count + 1)

    embedding_information.similarities[object_identifier] = _get_nearest_items_from_nearest_indexes(
        nearest_indexes, embedding_information
    )


def add_image_neighbors_to_cache(image_identifier: str, embedding_information: EmbeddingInformation):
    collection_id = embedding_information.keys_having_similarity[image_identifier]
    embedding = np.array([embedding_information.collections[collection_id]["embedding"]]).astype(np.float32)
    _, nearest_indexes = embedding_information.faiss_index.search(embedding, get_state().similarities_count + 1)

    embedding_information.similarities[image_identifier] = _get_nearest_items_from_nearest_indexes(
        nearest_indexes, embedding_information
    )
