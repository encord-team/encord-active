import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, overload

import pandas as pd
import streamlit as st
from encord.project_ontology.ontology import ObjectShape
from matplotlib import use
from pandas import Series
from pandera.typing import DataFrame
from streamlit.delta_generator import DeltaGenerator

from encord_active.app.common.components import (
    build_data_tags,
    multiselect_with_all_option,
)
from encord_active.app.common.components.annotator_statistics import (
    render_annotator_properties,
)
from encord_active.app.common.components.label_statistics import (
    render_dataset_properties,
)
from encord_active.app.common.components.paginator import render_pagination
from encord_active.app.common.components.similarities import (
    show_similar_classification_images,
    show_similar_images,
    show_similar_object_images,
)
from encord_active.app.common.components.slicer import render_df_slicer
from encord_active.app.common.components.tags.bulk_tagging_form import (
    BulkLevel,
    action_bulk_tags,
    bulk_tagging_form,
)
from encord_active.app.common.components.tags.individual_tagging import multiselect_tag
from encord_active.app.common.components.tags.tag_creator import tag_creator
from encord_active.app.common.page import Page
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
    OBJECT_KEYS_HAVING_SIMILARITIES,
    OBJECT_SIMILARITIES,
    QUESTION_HASH_TO_COLLECTION_INDEXES,
)
from encord_active.app.common.state_new import get_state, use_state
from encord_active.lib.charts.histogram import get_histogram
from encord_active.lib.common.image_utils import (
    load_or_fill_image,
    show_image_and_draw_polygons,
)
from encord_active.lib.embeddings.utils import (
    get_collections,
    get_collections_and_metadata,
    get_faiss_index_image,
    get_faiss_index_object,
    get_image_keys_having_similarities,
    get_object_keys_having_similarities,
)
from encord_active.lib.metrics.metric import (
    AnnotationType,
    EmbeddingType,
    MetricMetadata,
)
from encord_active.lib.metrics.utils import (
    MetricData,
    MetricSchema,
    MetricScope,
    get_annotator_level_info,
    load_metric_dataframe,
)


class ExplorerPage(Page):
    title = "🔎 Explorer"

    def sidebar_options(self, available_metrics: List[MetricData]):
        tag_creator()

        if not available_metrics:
            st.error("Your data has not been indexed. Make sure you have imported your data correctly.")
            st.stop()

        non_empty_metrics = [
            metric for metric in available_metrics if not load_metric_dataframe(metric, normalize=False).empty
        ]
        sorted_metrics = sorted(non_empty_metrics, key=lambda i: i.name)

        metric_names = list(map(lambda i: i.name, sorted_metrics))

        col1, col2, col3 = st.columns(3)
        selected_metric_name = col1.selectbox(
            "Select a metric to order your data by",
            metric_names,
            help="The data in the main view will be sorted by the selected metric. ",
        )

        if not selected_metric_name:
            return

        metric_idx = metric_names.index(selected_metric_name)
        selected_metric = sorted_metrics[metric_idx]
        get_state().selected_metric = selected_metric

        normalize = selected_metric.meta.get("score_normalization", get_state().normalize_metrics)
        df = load_metric_dataframe(selected_metric, normalize=normalize)

        if df.shape[0] <= 0:
            return

        class_set = sorted(list(df["object_class"].unique()))
        with col2:
            selected_classes = multiselect_with_all_option("Filter by class", class_set)

        is_class_selected = (
            df.shape[0] * [True] if "All" in selected_classes else df["object_class"].isin(selected_classes)
        )
        df_class_selected: DataFrame[MetricSchema] = df[is_class_selected]

        annotators = get_annotator_level_info(df_class_selected)
        annotator_set = sorted(annotators.keys())

        with col3:
            selected_annotators = multiselect_with_all_option(
                "Filter by annotator",
                annotator_set,
            )

        annotator_selected = (
            df_class_selected.shape[0] * [True]
            if "All" in selected_annotators
            else df_class_selected["annotator"].isin(selected_annotators)
        )

        self.row_col_settings_in_sidebar()
        # For now go the easy route and just filter the dataframe here
        return df_class_selected[annotator_selected]

    def build(self, selected_df: DataFrame[MetricSchema], metric_scope: MetricScope):
        selected_metric = get_state().selected_metric
        if not selected_metric:
            return

        st.markdown(f"# {self.title}")
        st.markdown(f"## {selected_metric.meta['title']}")
        st.markdown(selected_metric.meta["long_description"])

        if selected_df.empty:
            return

        with st.expander("Dataset Properties", expanded=True):
            render_dataset_properties(selected_df)
        with st.expander("Annotator Statistics", expanded=False):
            render_annotator_properties(selected_df)

        fill_data_quality_window(selected_df, metric_scope, selected_metric)


# TODO: move me to lib
def get_embedding_type(metric_title: str, annotation_type: Optional[List[Any]]) -> EmbeddingType:
    if (
        annotation_type is None
        or (len(annotation_type) == 1 and annotation_type[0] == str(AnnotationType.CLASSIFICATION.RADIO.value))
        or (metric_title in ["Frame object density", "Object Count"])
    ):  # TODO find a better way to filter these later because titles can change
        return EmbeddingType.CLASSIFICATION
    else:
        return EmbeddingType.OBJECT


def fill_data_quality_window(
    current_df: DataFrame[MetricSchema], metric_scope: MetricScope, selected_metric: MetricData
):
    meta = selected_metric.meta
    embedding_type = get_embedding_type(meta["title"], meta["annotation_type"])

    populate_embedding_information(embedding_type, meta)

    if (embedding_type == str(EmbeddingType.CLASSIFICATION.value)) and len(st.session_state[COLLECTIONS_IMAGES]) == 0:
        st.write("Image-level embedding file is not available for this project.")
        return
    if (embedding_type == str(EmbeddingType.OBJECT.value)) and len(st.session_state[COLLECTIONS_OBJECTS]) == 0:
        st.write("Object-level embedding file is not available for this project.")
        return

    n_cols = get_state().page_grid_settings.columns
    n_rows = get_state().page_grid_settings.rows

    metric = get_state().selected_metric
    if not metric:
        st.error("Metric not selected.")
        return

    chart = get_histogram(current_df, "score", metric.name)
    st.altair_chart(chart, use_container_width=True)
    subset = render_df_slicer(current_df, "score")

    st.write(f"Interval contains {subset.shape[0]} of {current_df.shape[0]} annotations")

    paginated_subset = render_pagination(subset, n_cols, n_rows, "score")

    form = bulk_tagging_form(metric_scope)

    if form and form.submitted:
        df = paginated_subset if form.level == BulkLevel.PAGE else subset
        action_bulk_tags(df, form.tags, form.action)

    if len(paginated_subset) == 0:
        st.error("No data in selected interval")
    else:
        cols: List = []
        similarity_expanders = []
        for i, (_, row) in enumerate(paginated_subset.iterrows()):
            if not cols:
                cols = list(st.columns(n_cols))
                similarity_expanders.append(st.expander("Similarities", expanded=True))

            with cols.pop(0):
                build_card(embedding_type, i, row, similarity_expanders, metric_scope, metric)


# @dataclass
# class EmbeddingInformation:
#     embdedding_type: EmbeddingType
#     collection_images: List[LabelEmbedding]
#     question_hash_to_collection_indexes: Dict[str, Any]
#     image_keys_having_similarity: Dict[str, Any]
#


def populate_embedding_information(embedding_type: EmbeddingType, meta: MetricMetadata):
    embeddings_dir = get_state().project_paths.embeddings

    if embedding_type == EmbeddingType.CLASSIFICATION:
        if meta["title"] == "Image-level Annotation Quality":
            collections, question_hash_to_collection_indexes = get_collections_and_metadata(
                "cnn_classifications.pkl", embeddings_dir
            )
            st.session_state[COLLECTIONS_IMAGES] = collections
            st.session_state[QUESTION_HASH_TO_COLLECTION_INDEXES] = question_hash_to_collection_indexes
            st.session_state[IMAGE_KEYS_HAVING_SIMILARITIES] = get_image_keys_having_similarities(collections)
            st.session_state[FAISS_INDEX_IMAGE] = get_faiss_index_image(
                collections, question_hash_to_collection_indexes
            )
            st.session_state[CURRENT_INDEX_HAS_ANNOTATION] = True

            if IMAGE_SIMILARITIES not in st.session_state:
                st.session_state[IMAGE_SIMILARITIES] = {}
                for question_hash in st.session_state[QUESTION_HASH_TO_COLLECTION_INDEXES].keys():
                    st.session_state[IMAGE_SIMILARITIES][question_hash] = {}
        else:
            collections = get_collections("cnn_classifications.pkl", embeddings_dir)
            st.session_state[COLLECTIONS_IMAGES] = collections
            st.session_state[IMAGE_KEYS_HAVING_SIMILARITIES] = get_image_keys_having_similarities(collections)
            st.session_state[FAISS_INDEX_IMAGE_NO_LABEL] = get_faiss_index_object(collections)
            st.session_state[CURRENT_INDEX_HAS_ANNOTATION] = False

            if IMAGE_SIMILARITIES_NO_LABEL not in st.session_state:
                st.session_state[IMAGE_SIMILARITIES_NO_LABEL] = {}

    elif embedding_type == EmbeddingType.OBJECT:
        collections = get_collections("cnn_objects.pkl", embeddings_dir)
        st.session_state[COLLECTIONS_OBJECTS] = collections
        st.session_state[OBJECT_KEYS_HAVING_SIMILARITIES] = get_object_keys_having_similarities(collections)
        st.session_state[FAISS_INDEX_OBJECT] = get_faiss_index_object(collections)
        st.session_state[CURRENT_INDEX_HAS_ANNOTATION] = True

        if OBJECT_SIMILARITIES not in st.session_state:
            st.session_state[OBJECT_SIMILARITIES] = {}


def build_card(
    embedding_type: EmbeddingType,
    card_no: int,
    row: Series,
    similarity_expanders: list[DeltaGenerator],
    metric_scope: MetricScope,
    metric: MetricData,
):
    """
    Builds each sub card (the content displayed for each row in a csv file).
    """
    data_dir = get_state().project_paths.data

    if embedding_type == EmbeddingType.CLASSIFICATION:
        button_name = "show similar images"
        if metric.meta["title"] == "Image-level Annotation Quality":
            image = load_or_fill_image(row, data_dir)
            similarity_callback = show_similar_classification_images
        else:
            if metric.meta["annotation_type"] is None:
                image = load_or_fill_image(row, data_dir)
            else:
                image = show_image_and_draw_polygons(row, data_dir)
            similarity_callback = show_similar_images
    elif embedding_type == EmbeddingType.OBJECT:
        image = show_image_and_draw_polygons(row, data_dir)
        button_name = "show similar objects"
        similarity_callback = show_similar_object_images
    else:
        st.write(f"{embedding_type.value} card type is not defined in EmbeddingTypes")
        return

    st.image(image)
    multiselect_tag(row, "explorer", metric_scope)

    target_expander = similarity_expanders[card_no // get_state().page_grid_settings.columns]

    st.button(
        str(button_name),
        key=f"similarity_button_{row['identifier']}",
        on_click=similarity_callback,
        args=(row, target_expander),
    )

    # === Write scores and link to editor === #
    tags_row = row.copy()

    if "object_class" in tags_row and not pd.isna(tags_row["object_class"]):
        tags_row["label_class_name"] = tags_row["object_class"]
        tags_row.drop("object_class")
    tags_row[metric.name] = tags_row["score"]
    build_data_tags(tags_row, metric.name)

    if not pd.isnull(row["description"]):
        # Hacky way for now (with incorrect rounding)
        description = re.sub(r"(\d+\.\d{0,3})\d*", r"\1", row["description"])
        st.write(f"Description: {description}")
