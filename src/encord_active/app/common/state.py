from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st
from pandera.typing import DataFrame

from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tag, Tags
from encord_active.lib.metrics.utils import MetricData
from encord_active.lib.model_predictions.reader import LabelSchema, OntologyObjectJSON
from encord_active.lib.project import ProjectFileStructure

GLOBAL_STATE = "global_state"


@dataclass
class MetricNames:
    predictions: Dict[str, MetricData] = field(default_factory=dict)
    selected_predicion: Optional[str] = None
    labels: Dict[str, MetricData] = field(default_factory=dict)
    selected_label: Optional[str] = None


@dataclass
class PredictionsState:
    decompose_classes = False
    metric_datas = MetricNames()
    all_classes: Dict[str, OntologyObjectJSON] = field(default_factory=dict)
    selected_classes: Dict[str, OntologyObjectJSON] = field(default_factory=dict)
    labels: Optional[DataFrame[LabelSchema]] = None
    nbins = 50


@dataclass
class PageGridSettings:
    columns: int = 4
    rows: int = 5


@dataclass
class State:
    """This is not intended for usage, please use the `get_state` constant instead."""

    project_paths: ProjectFileStructure
    all_tags: List[Tag]
    merged_metrics: pd.DataFrame
    ignore_frames_without_predictions = False
    iou_threshold = 0.5
    selected_metric: Optional[MetricData] = None
    page_grid_settings = PageGridSettings()
    normalize_metrics = False
    predictions = PredictionsState()

    @classmethod
    def init(cls, project_dir: Path):
        if GLOBAL_STATE in st.session_state:
            return

        st.session_state[GLOBAL_STATE] = State(
            project_paths=ProjectFileStructure(project_dir),
            merged_metrics=MergedMetrics().all(),
            all_tags=Tags().all(),
        )


def get_state() -> State:
    return st.session_state.get(GLOBAL_STATE)  # type: ignore


# EVERYTHING BELOW SHOULD BE DEPRACATED

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
K_NEAREST_NUM = "k_nearest_num"


def setdefault(key: str, fn: Callable, *args, **kwargs) -> Any:
    if not key in st.session_state:
        st.session_state[key] = fn(*args, **kwargs)
    return st.session_state.get(key)
