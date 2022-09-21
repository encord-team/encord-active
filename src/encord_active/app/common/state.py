from pathlib import Path

import streamlit as st

# CONSTANTS
PROJECT_CACHE_FILE = Path.home() / ".encord_assertions" / "current_project_dir.txt"

# DATABASE
DB_FILE_NAME = "sqlite.db"

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
PREDICTIONS_LABEL_METRIC = "predictions_label_metric"
PREDICTIONS_DECOMPOSE_CLASSES = "predictions_decompose_classes"
PREDICTIONS_METRIC = "predictions_metric"
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


def set_project_dir(project_dir: str) -> bool:
    _project_dir = Path(project_dir).expanduser().absolute()

    if not _project_dir.is_dir():
        return False
    else:
        PROJECT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with PROJECT_CACHE_FILE.open("w", encoding="utf-8") as f:
            f.write(_project_dir.as_posix())
        st.session_state.project_dir = _project_dir
        return True


def populate_session_state():
    if "project_dir" not in st.session_state:
        # Try using a cached one
        if PROJECT_CACHE_FILE.is_file():
            with PROJECT_CACHE_FILE.open("r", encoding="utf-8") as f:
                st.session_state.project_dir = Path(f.readline())

    st.session_state.metric_dir = st.session_state.project_dir / "metrics"
    st.session_state.embeddings_dir = st.session_state.project_dir / "embeddings"
    st.session_state.predictions_dir = st.session_state.project_dir / "predictions"
    st.session_state.data_dir = st.session_state.project_dir / "data"
    st.session_state.ontology_file = st.session_state.project_dir / "ontology.json"
    st.session_state.db_path = st.session_state.project_dir / DB_FILE_NAME
