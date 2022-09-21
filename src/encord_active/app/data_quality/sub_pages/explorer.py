import re
from typing import List

import altair as alt
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from natsort import natsorted
from pandas import Series
from streamlit.delta_generator import DeltaGenerator

import encord_active.app.common.state as state
from encord_active.app.common import embedding_utils
from encord_active.app.common.components import (
    build_data_tags,
    multiselect_with_all_option,
)
from encord_active.app.common.components.bulk_tagging_form import (
    BulkLevel,
    action_bulk_tags,
    bulk_tagging_form,
)
from encord_active.app.common.components.individual_tagging import (
    multiselect_tag,
    tag_creator,
)
from encord_active.app.common.metric import MetricData, load_metric
from encord_active.app.common.page import Page
from encord_active.app.common.utils import (
    build_pagination,
    get_df_subset,
    load_merged_df,
)
from encord_active.app.data_quality.common import (
    load_or_fill_image,
    show_image_and_draw_polygons,
)
from encord_active.lib.common.metric import AnnotationType, EmbeddingType


class ExplorerPage(Page):
    title = "🔎 Explorer"

    def sidebar_options(self, available_metrics: List[MetricData]):
        load_merged_df()
        tag_creator()

        if not available_metrics:
            st.error("Your data has not been indexed. Make sure you have imported your data correctly.")
            st.stop()

        non_empty_metrics = [metric for metric in available_metrics if not load_metric(metric, normalize=False).empty]
        sorted_metrics = sorted(non_empty_metrics, key=lambda i: i.name)

        metric_names = list(map(lambda i: i.name, sorted_metrics))

        col1, col2, col3 = st.columns(3)
        selected_metric_name = col1.selectbox(
            "Select a metric to order your data by",
            metric_names,
            help="The data in the main view will be sorted by the selected metric. ",
        )

        metric_idx = metric_names.index(selected_metric_name)
        st.session_state[state.DATA_PAGE_METRIC] = sorted_metrics[metric_idx]
        st.session_state[state.DATA_PAGE_METRIC_NAME] = selected_metric_name

        if state.NORMALIZATION_STATUS not in st.session_state:
            st.session_state[state.NORMALIZATION_STATUS] = st.session_state[state.DATA_PAGE_METRIC].meta.get(
                state.METRIC_METADATA_SCORE_NORMALIZATION, True
            )  # If there is no information on the meta file, just normalize (its probability is higher)

        df = load_metric(
            st.session_state[state.DATA_PAGE_METRIC], normalize=st.session_state[state.NORMALIZATION_STATUS]
        )

        if df.shape[0] > 0:
            class_set = sorted(list(df["object_class"].unique()))
            with col2:
                selected_classes = multiselect_with_all_option("Filter by class", class_set, key=state.DATA_PAGE_CLASS)

            is_class_selected = (
                df.shape[0] * [True] if "All" in selected_classes else df["object_class"].isin(selected_classes)
            )
            df_class_selected = df[is_class_selected]

            annotators = get_annotator_level_info(df_class_selected)
            annotator_set = sorted(annotators["annotator"])

            with col3:
                selected_annotators = multiselect_with_all_option(
                    "Filter by annotator",
                    annotator_set,
                    key=state.DATA_PAGE_ANNOTATOR,
                )

            annotator_selected = (
                df_class_selected.shape[0] * [True]
                if "All" in selected_annotators
                else df_class_selected["annotator"].isin(selected_annotators)
            )

            self.row_col_settings_in_sidebar()
            # For now go the easy route and just filter the dataframe here
            return df_class_selected[annotator_selected]

    def build(self, selected_df: pd.DataFrame):
        st.markdown(f"# {self.title}")
        meta = st.session_state[state.DATA_PAGE_METRIC].meta
        st.markdown(f"## {meta['title']}")
        st.markdown(meta["long_description"])

        if selected_df.empty:
            return

        fill_dataset_properties_window(selected_df)
        fill_annotator_properties_window(selected_df)
        fill_data_quality_window(selected_df)


def get_annotator_level_info(df: pd.DataFrame) -> dict[str, list]:
    annotator_set = natsorted(list(df["annotator"].unique()))
    annotators: dict[str, list] = {"annotator": annotator_set, "total annotation": [], "score": []}

    for annotator in annotator_set:
        annotators["total annotation"].append(df[df["annotator"] == annotator].shape[0])
        annotators["score"].append(df[df["annotator"] == annotator]["score"].mean())

    return annotators


def fill_dataset_properties_window(current_df: pd.DataFrame):
    dataset_expander = st.expander("Dataset Properties", expanded=True)
    dataset_columns = dataset_expander.columns(3)

    cls_set = natsorted(list(current_df["object_class"].unique()))

    dataset_columns[0].metric("Number of labels", current_df.shape[0])
    dataset_columns[1].metric("Number of classes", len(cls_set))
    dataset_columns[2].metric("Number of images", get_unique_data_units_size(current_df))

    if len(cls_set) > 1:
        classes = {}
        for cls in cls_set:
            classes[cls] = (current_df["object_class"] == cls).sum()

        source = pd.DataFrame({"class": list(classes.keys()), "count": list(classes.values())})

        fig = px.bar(source, x="class", y="count")
        fig.update_layout(title_text="Distribution of the classes", title_x=0.5, title_font_size=20)
        dataset_expander.plotly_chart(fig, use_container_width=True)


def fill_annotator_properties_window(current_df: pd.DataFrame):
    annotators = get_annotator_level_info(current_df)
    if not (len(annotators["annotator"]) == 1 and (not isinstance(annotators["annotator"][0], str))):
        annotator_expander = st.expander("Annotator Statistics", expanded=False)

        annotator_columns = annotator_expander.columns([2, 2])

        # 1. Pie Chart
        annotator_columns[0].markdown(
            "<h5 style='text-align: center; color: black;'>Distribution of the annotations</h1>", unsafe_allow_html=True
        )
        source = pd.DataFrame(
            {
                "annotator": annotators["annotator"],
                "total": annotators["total annotation"],
                "score": [f"{score:.3f}" for score in annotators["score"]],
            }
        )

        fig = px.pie(source, values="total", names="annotator", hover_data=["score"])
        # fig.update_layout(title_text="Distribution of the annotations", title_x=0.5, title_font_size=20)

        annotator_columns[0].plotly_chart(fig, use_container_width=True)

        # 2. Table View
        annotator_columns[1].markdown(
            "<h5 style='text-align: center; color: black;'>Detailed annotator statistics</h1>", unsafe_allow_html=True
        )

        annotators["annotator"].append("all")
        annotators["total annotation"].append(current_df.shape[0])

        df_mean_score = current_df["score"].mean()
        annotators["score"].append(df_mean_score)
        deviations = 100 * ((np.array(annotators["score"]) - df_mean_score) / df_mean_score)
        annotators["deviations"] = deviations
        annotators_df = pd.DataFrame.from_dict(annotators)

        def _format_deviation(val):
            return f"{val:.1f}%"

        def _format_score(val):
            return f"{val:.3f}"

        def _color_red_or_green(val):
            color = "red" if val < 0 else "green"
            return f"color: {color}"

        def make_pretty(styler):
            styler.format(_format_deviation, subset=["deviations"])
            styler.format(_format_score, subset=["score"])
            styler.applymap(_color_red_or_green, subset=["deviations"])
            return styler

        annotator_columns[1].dataframe(annotators_df.style.pipe(make_pretty), use_container_width=True)


def fill_data_quality_window(current_df: pd.DataFrame):
    annotation_type = st.session_state[state.DATA_PAGE_METRIC].meta.get("annotation_type")
    if (
        (annotation_type is None)
        or (len(annotation_type) == 1 and annotation_type[0] == str(AnnotationType.CLASSIFICATION.RADIO.value))
        or (
            st.session_state[state.DATA_PAGE_METRIC].meta.get("title")
            in [
                "Frame object density",
                "Object Count",
            ]
        )
    ):  # TODO find a better way to filter these later because titles can change
        embedding_type = str(EmbeddingType.CLASSIFICATION.value)
    else:
        embedding_type = str(EmbeddingType.OBJECT.value)

    populate_embedding_information(embedding_type)

    if (embedding_type == str(EmbeddingType.CLASSIFICATION.value)) and len(
        st.session_state[state.COLLECTIONS_IMAGES]
    ) == 0:
        st.write("Image-level embedding file is not available for this project.")
        return
    if (embedding_type == str(EmbeddingType.OBJECT.value)) and len(st.session_state[state.COLLECTIONS_OBJECTS]) == 0:
        st.write("Object-level embedding file is not available for this project.")
        return

    n_cols = int(st.session_state[state.MAIN_VIEW_COLUMN_NUM])
    n_rows = int(st.session_state[state.MAIN_VIEW_ROW_NUM])

    chart = get_histogram(current_df)
    st.altair_chart(chart, use_container_width=True)
    subset = get_df_subset(current_df, "score")

    st.write(f"Interval contains {subset.shape[0]} of {current_df.shape[0]} annotations")

    paginated_subset = build_pagination(subset, n_cols, n_rows, "score")

    form = bulk_tagging_form()

    if form and form.submitted:
        df = paginated_subset if form.level == BulkLevel.PAGE else subset
        action_bulk_tags(df, form.tags, form.action)

    if len(paginated_subset) == 0:
        st.error("No data in selected interval")
    else:
        cols: List = []
        similarity_expanders = []
        for i, (row_no, row) in enumerate(paginated_subset.iterrows()):
            if not cols:
                cols = list(st.columns(n_cols))
                similarity_expanders.append(st.expander("Similarities", expanded=True))

            with cols.pop(0):
                build_card(embedding_type, i, row, similarity_expanders)


def populate_embedding_information(embedding_type: str):
    if embedding_type == EmbeddingType.CLASSIFICATION.value:
        if st.session_state[state.DATA_PAGE_METRIC].meta.get("title") == "Image-level Annotation Quality":
            collections, question_hash_to_collection_indexes = embedding_utils.get_collections_and_metadata(
                "cnn_classifications.pkl"
            )
            st.session_state[state.COLLECTIONS_IMAGES] = collections
            st.session_state[state.QUESTION_HASH_TO_COLLECTION_INDEXES] = question_hash_to_collection_indexes
            st.session_state[state.IMAGE_KEYS_HAVING_SIMILARITIES] = embedding_utils.get_image_keys_having_similarities(
                collections
            )
            st.session_state[state.FAISS_INDEX_IMAGE] = embedding_utils.get_faiss_index_image(collections)
            st.session_state[state.CURRENT_INDEX_HAS_ANNOTATION] = True

            if state.IMAGE_SIMILARITIES not in st.session_state:
                st.session_state[state.IMAGE_SIMILARITIES] = {}
                for question_hash in st.session_state[state.QUESTION_HASH_TO_COLLECTION_INDEXES].keys():
                    st.session_state[state.IMAGE_SIMILARITIES][question_hash] = {}
        else:
            collections = embedding_utils.get_collections("cnn_classifications.pkl")
            st.session_state[state.COLLECTIONS_IMAGES] = collections
            st.session_state[state.IMAGE_KEYS_HAVING_SIMILARITIES] = embedding_utils.get_image_keys_having_similarities(
                collections
            )
            st.session_state[state.FAISS_INDEX_IMAGE_NO_LABEL] = embedding_utils.get_faiss_index_object(
                collections, state.FAISS_INDEX_IMAGE_NO_LABEL
            )
            st.session_state[state.CURRENT_INDEX_HAS_ANNOTATION] = False

            if state.IMAGE_SIMILARITIES_NO_LABEL not in st.session_state:
                st.session_state[state.IMAGE_SIMILARITIES_NO_LABEL] = {}

    elif embedding_type == EmbeddingType.OBJECT.value:
        collections = embedding_utils.get_collections("cnn_objects.pkl")
        st.session_state[state.COLLECTIONS_OBJECTS] = collections
        st.session_state[state.OBJECT_KEYS_HAVING_SIMILARITIES] = embedding_utils.get_object_keys_having_similarities(
            collections
        )
        st.session_state[state.FAISS_INDEX_OBJECT] = embedding_utils.get_faiss_index_object(
            collections, state.FAISS_INDEX_OBJECT
        )
        st.session_state[state.CURRENT_INDEX_HAS_ANNOTATION] = True

        if state.OBJECT_SIMILARITIES not in st.session_state:
            st.session_state[state.OBJECT_SIMILARITIES] = {}


def build_card(card_type: str, card_no: int, row: Series, _similarity_expanders: list[DeltaGenerator]):
    """
    Builds each sub card (the content displayed for each row in a csv file).
    """

    if card_type == EmbeddingType.CLASSIFICATION.value:

        button_name = "show similar images"
        if st.session_state[state.DATA_PAGE_METRIC].meta.get("title") == "Image-level Annotation Quality":
            image = load_or_fill_image(row)
            similarity_callback = show_similar_classification_images
        else:
            if st.session_state[state.DATA_PAGE_METRIC].meta.get("annotation_type") is None:
                image = load_or_fill_image(row)
            else:
                image = show_image_and_draw_polygons(row)
            similarity_callback = show_similar_images
    elif card_type == EmbeddingType.OBJECT.value:
        image = show_image_and_draw_polygons(row)
        button_name = "show similar objects"
        similarity_callback = show_similar_object_images

    else:
        st.write(f"{card_type} card type is not defined in EmbeddingTypes")
        return

    st.image(image)
    multiselect_tag(row, "explorer")

    target_expander = _similarity_expanders[card_no // st.session_state[state.MAIN_VIEW_COLUMN_NUM]]

    st.button(
        str(button_name),
        key=f"similarity_button_{row['identifier']}",
        on_click=similarity_callback,
        args=(row, target_expander),
    )

    # === Write scores and link to editor === #
    tags_row = row.copy()
    metric_name = st.session_state[state.DATA_PAGE_METRIC_NAME]
    if "object_class" in tags_row and not pd.isna(tags_row["object_class"]):
        tags_row["label_class_name"] = tags_row["object_class"]
        tags_row.drop("object_class")
    tags_row[metric_name] = tags_row["score"]
    build_data_tags(tags_row, metric_name)

    if not pd.isnull(row["description"]):
        # Hacky way for now (with incorrect rounding)
        description = re.sub(r"(\d+\.\d{0,3})\d*", r"\1", row["description"])
        st.write(f"Description: {description}")


def get_histogram(current_df: pd.DataFrame):
    # TODO: Unify with app/model_assertions/sub_pages/__init__.py:SamplesPage.get_histogram
    metric_name = st.session_state[state.DATA_PAGE_METRIC_NAME]
    if metric_name:
        title_suffix = f" - {metric_name}"
    else:
        metric_name = "Score"  # Used for plotting

    bar_chart = (
        alt.Chart(current_df, title=f"Data distribution{title_suffix}")
        .mark_bar()
        .encode(
            alt.X("score:Q", bin=alt.Bin(maxbins=100), title=metric_name),
            alt.Y("count()", title="Num. samples"),
            tooltip=[
                alt.Tooltip("score:Q", title=metric_name, format=",.3f", bin=True),
                alt.Tooltip("count():Q", title="Num. samples", format="d"),
            ],
        )
        .properties(height=200)
    )
    return bar_chart


def show_similar_classification_images(row: Series, expander: DeltaGenerator):
    feature_hash = row["identifier"].split("_")[-1]

    if row["identifier"] not in st.session_state[state.IMAGE_SIMILARITIES][feature_hash].keys():
        embedding_utils.add_labeled_image_neighbors_to_cache(row["identifier"], feature_hash)

    nearest_images = st.session_state[state.IMAGE_SIMILARITIES][feature_hash][row["identifier"]]

    division = 4
    column_id = 0

    for nearest_image in nearest_images:
        if column_id == 0:
            st_columns = expander.columns(division)

        image = load_or_fill_image(nearest_image["key"])

        st_columns[column_id].image(image)
        st_columns[column_id].write(f"Annotated as `{nearest_image['name']}`")
        column_id += 1
        column_id = column_id % division


def show_similar_images(row: Series, expander: DeltaGenerator):
    image_identifier = "_".join(row["identifier"].split("_")[:3])

    if image_identifier not in st.session_state[state.IMAGE_SIMILARITIES_NO_LABEL].keys():
        embedding_utils.add_image_neighbors_to_cache(image_identifier)

    nearest_images = st.session_state[state.IMAGE_SIMILARITIES_NO_LABEL][image_identifier]

    division = 4
    column_id = 0

    for nearest_image in nearest_images:
        if column_id == 0:
            st_columns = expander.columns(division)

        image = load_or_fill_image(nearest_image["key"])

        st_columns[column_id].image(image)
        st_columns[column_id].write(f"Annotated as `{nearest_image['name']}`")
        column_id += 1
        column_id = column_id % division


def show_similar_object_images(row: Series, expander: DeltaGenerator):
    object_identifier = "_".join(row["identifier"].split("_")[:4])

    if object_identifier not in st.session_state[state.OBJECT_KEYS_HAVING_SIMILARITIES]:
        expander.write("Similarity search is not available for this object.")
        return

    if object_identifier not in st.session_state[state.OBJECT_SIMILARITIES].keys():
        embedding_utils.add_object_neighbors_to_cache(object_identifier)

    nearest_images = st.session_state[state.OBJECT_SIMILARITIES][object_identifier]

    division = 4
    column_id = 0

    for nearest_image in nearest_images:
        if column_id == 0:
            st_columns = expander.columns(division)

        image = show_image_and_draw_polygons(nearest_image["key"])

        st_columns[column_id].image(image)
        st_columns[column_id].write(f"Annotated as `{nearest_image['name']}`")
        column_id += 1
        column_id = column_id % division


def get_unique_data_units_size(current_df: pd.DataFrame):
    data_units = set()
    identifiers = current_df["identifier"]
    for identifier in identifiers:
        key_components = identifier.split("_")
        data_units.add(key_components[0] + "_" + key_components[1])

    return len(data_units)
