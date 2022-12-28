from typing import Any, Callable

import streamlit as st

# SIMILARITY KEYS
OBJECT_KEYS_HAVING_SIMILARITIES = "object_keys_having_similarities"
IMAGE_KEYS_HAVING_SIMILARITIES = "image_keys_having_similarity"
OBJECT_SIMILARITIES = "object_similarities"
IMAGE_SIMILARITIES = "image_similarities"
IMAGE_SIMILARITIES_NO_LABEL = "image_similarities_no_label"
FAISS_INDEX_OBJECT = "faiss_index_object"
FAISS_INDEX_IMAGE = "faiss_index_image"
FAISS_INDEX_IMAGE_NO_LABEL = "faiss_index_image_no_label"
CURRENT_INDEX_HAS_ANNOTATION = "current_index_has_annotation"
QUESTION_HASH_TO_COLLECTION_INDEXES = "question_hash_to_collection_indexes"
COLLECTIONS_IMAGES = "collections_images"
COLLECTIONS_OBJECTS = "collections_objects"


# DATA QUALITY PAGE
DATA_PAGE_ANNOTATOR = "data_page_annotator_selection"  # annotator


# PREDICTIONS PAGE
PREDICTIONS_DECOMPOSE_CLASSES = "predictions_decompose_classes"
PREDICTIONS_FULL_CLASS_IDX = "full_class_idx"
PREDICTIONS_GT_MATCHED = "gt_matched"
PREDICTIONS_LABELS = "labels"
PREDICTIONS_LABEL_METRIC = "predictions_label_metric"
PREDICTIONS_LABEL_METRIC_NAMES = "label_metric_names"
PREDICTIONS_METRIC = "predictions_metric"
PREDICTIONS_METRIC_META = "metric_meta"
PREDICTIONS_METRIC_NAMES = "prediction_metric_names"
PREDICTIONS_MODEL_PREDICTIONS = "model_predictions"
PREDICTIONS_NBINS = "predictions_nbins"

# TILING & PAGINATION
MAIN_VIEW_COLUMN_NUM = "main_view_column_num"
MAIN_VIEW_ROW_NUM = "main_view_row_num"
K_NEAREST_NUM = "k_nearest_num"

METRIC_VIEW_PAGE_NUMBER = "metric_view_page_number"
FALSE_NEGATIVE_VIEW_PAGE_NUMBER = "false_negative_view_page_number"

NORMALIZATION_STATUS = "normalization_status"
METRIC_METADATA_SCORE_NORMALIZATION = "score_normalization"

# Export page
NUMBER_OF_PARTITIONS = "number_of_partitions"
ACTION_PAGE_PREVIOUS_FILTERED_NUM = "action_page_previous_filtered"


def setdefault(key: str, fn: Callable, *args, **kwargs) -> Any:
    if not key in st.session_state:
        st.session_state[key] = fn(*args, **kwargs)
    return st.session_state.get(key)
