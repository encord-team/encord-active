from pathlib import Path
from typing import Any, Callable

import streamlit as st

from encord_active.lib.common.project_file_structure import ProjectFileStructure
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tags

# CONSTANTS
PROJECT_CACHE_FILE = Path.home() / ".encord_quality" / "current_project_dir.txt"

# STATE VARIABLE KEYS
CLASS_SELECTION = "class_selection"
IGNORE_FRAMES_WO_PREDICTIONS = "ignore_frames_wo_predictions"
IOU_THRESHOLD = "iou_threshold_scaled"  # After normalization
IOU_THRESHOLD_ = "iou_threshold_"  # Before normalization
MERGED_DATAFRAME = "merged_dataframe"
ALL_TAGS = "all_tags"

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
DATA_PAGE_METRIC = "data_page_metric"  # metric
DATA_PAGE_METRIC_NAME = "data_page_metric_name"  # metric name
DATA_PAGE_CLASS = "data_page_class_selection"  # class
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
ACTION_PAGE_CLONE_BUTTON = "action_page_clone_button"
ACTION_PAGE_PREVIOUS_FILTERED_NUM = "action_page_previous_filtered"


def populate_session_state():
    project_file_structure = ProjectFileStructure(st.session_state.project_dir)
    st.session_state.project_file_structure = project_file_structure
    st.session_state.metric_dir = project_file_structure.metrics
    st.session_state.embeddings_dir = project_file_structure.embeddings
    st.session_state.predictions_dir = project_file_structure.predictions
    st.session_state.data_dir = project_file_structure.data
    st.session_state.ontology_file = project_file_structure.ontology
    st.session_state.db_path = project_file_structure.db

    DBConnection.set_project_path(st.session_state.project_dir)

    if MERGED_DATAFRAME not in st.session_state:
        st.session_state[MERGED_DATAFRAME] = MergedMetrics().all()

    if ALL_TAGS not in st.session_state:
        st.session_state[ALL_TAGS] = Tags().all()


def setdefault(key: str, fn: Callable, *args, **kwargs) -> Any:
    if not key in st.session_state:
        st.session_state[key] = fn(*args, **kwargs)
    return st.session_state.get(key)
