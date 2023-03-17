from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import streamlit as st
from pandera.typing import DataFrame

from encord_active.lib.common.image_utils import ObjectDrawingConfigurations
from encord_active.lib.dataset.outliers import MetricsSeverity
from encord_active.lib.dataset.summary_utils import AnnotationStatistics
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tag, Tags
from encord_active.lib.embeddings.utils import Embedding2DSchema
from encord_active.lib.metrics.metric import EmbeddingType
from encord_active.lib.metrics.utils import MetricData
from encord_active.lib.model_predictions.reader import LabelSchema, OntologyObjectJSON
from encord_active.lib.model_predictions.writer import OntologyClassificationJSON
from encord_active.lib.project import ProjectFileStructure

GLOBAL_STATE = "global_state"


@dataclass
class MetricNames:
    predictions: Dict[str, MetricData] = field(default_factory=dict)
    selected_prediction: Optional[str] = None
    labels: Dict[str, MetricData] = field(default_factory=dict)
    selected_label: Optional[str] = None


@dataclass
class PredictionsState:
    decompose_classes = False
    metric_datas = MetricNames()
    metric_datas_classification = MetricNames()
    all_classes_objects: Dict[str, OntologyObjectJSON] = field(default_factory=dict)
    all_classes_classifications: Dict[str, OntologyClassificationJSON] = field(default_factory=dict)
    selected_classes_objects: Dict[str, OntologyObjectJSON] = field(default_factory=dict)
    selected_classes_classifications: Dict[str, OntologyObjectJSON] = field(default_factory=dict)
    labels: Optional[DataFrame[LabelSchema]] = None
    nbins: int = 50


@dataclass
class PageGridSettings:
    columns: int = 4
    rows: int = 5


@dataclass
class State:
    """
    Use this only for getting default values, otherwise use `get_state()`
    To get the default `iou_threshold` we would call `State.iou_threshold`
    and to get/set the current value we would call `get_state().iou_threshold.`
    """

    project_paths: ProjectFileStructure
    all_tags: List[Tag]
    merged_metrics: pd.DataFrame
    ignore_frames_without_predictions = False
    iou_threshold = 0.5
    selected_metric: Optional[MetricData] = None
    page_grid_settings = PageGridSettings()
    predictions = PredictionsState()
    similarities_count = 8
    image_sizes: Optional[np.ndarray] = None
    annotation_sizes: Optional[AnnotationStatistics] = None
    metrics_data_summary: Optional[MetricsSeverity] = None
    metrics_label_summary: Optional[MetricsSeverity] = None
    object_drawing_configurations: ObjectDrawingConfigurations = field(
        default_factory=lambda: ObjectDrawingConfigurations()
    )
    reduced_embeddings: dict[EmbeddingType, Optional[DataFrame[Embedding2DSchema]]] = field(default_factory=dict)

    @classmethod
    def init(cls, project_dir: Path):
        if GLOBAL_STATE not in st.session_state or project_dir != get_state().project_paths.project_dir:
            st.session_state[GLOBAL_STATE] = State(
                project_paths=ProjectFileStructure(project_dir),
                merged_metrics=MergedMetrics().all(),
                all_tags=Tags().all(),
            )

    @classmethod
    def clear(cls) -> None:
        st.session_state.pop(GLOBAL_STATE, None)


def get_state() -> State:
    return st.session_state.get(GLOBAL_STATE)  # type: ignore


def refresh(clear_state=False):
    if clear_state:
        State.clear()
    st.experimental_rerun()
