from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import streamlit as st
from pandera.typing import DataFrame

from encord_active.lib.common.image_utils import ObjectDrawingConfigurations
from encord_active.lib.dataset.outliers import MetricsSeverity
from encord_active.lib.dataset.summary_utils import AnnotationStatistics
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tag, Tags
from encord_active.lib.embeddings.utils import Embedding2DSchema
from encord_active.lib.metrics.metric import EmbeddingType
from encord_active.lib.metrics.utils import MetricData
from encord_active.lib.model_predictions.reader import LabelSchema, OntologyObjectJSON
from encord_active.lib.model_predictions.writer import OntologyClassificationJSON
from encord_active.lib.project import ProjectFileStructure


class StateKey(str, Enum):
    GLOBAL = "GLOBAL"
    SCOPED = "SCOPED"
    MEMO = "MEMO"


@dataclass
class MetricNames:
    predictions: Dict[str, MetricData] = field(default_factory=dict)
    selected_prediction: Optional[str] = None
    labels: Dict[str, MetricData] = field(default_factory=dict)
    selected_label: Optional[str] = None


@dataclass
class PredictionsState:
    decompose_classes = False
    metric_datas: MetricNames = field(default_factory=MetricNames)
    metric_datas_classification: MetricNames = field(default_factory=MetricNames)
    all_classes_objects: Dict[str, OntologyObjectJSON] = field(default_factory=dict)
    all_classes_classifications: Dict[str, OntologyClassificationJSON] = field(default_factory=dict)
    selected_classes_objects: Dict[str, OntologyObjectJSON] = field(default_factory=dict)
    selected_classes_classifications: Dict[str, OntologyClassificationJSON] = field(default_factory=dict)
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
    refresh_projects: Callable[[], Any]
    all_tags: List[Tag]
    merged_metrics: pd.DataFrame
    ignore_frames_without_predictions = False
    iou_threshold = 0.5
    selected_metric: Optional[MetricData] = None
    page_grid_settings: PageGridSettings = field(default_factory=PageGridSettings)
    predictions: PredictionsState = field(default_factory=PredictionsState)
    similarities_count = 8
    annotation_sizes: Optional[AnnotationStatistics] = None
    metrics_data_summary: Optional[MetricsSeverity] = None
    metrics_label_summary: Optional[MetricsSeverity] = None
    object_drawing_configurations: ObjectDrawingConfigurations = field(default_factory=ObjectDrawingConfigurations)
    reduced_embeddings: dict[EmbeddingType, Optional[DataFrame[Embedding2DSchema]]] = field(default_factory=dict)

    @classmethod
    def init(cls, project_dir: Path, refresh_projects: Callable[[], Any]):
        if (
            st.session_state.get(StateKey.GLOBAL) is None
            or project_dir != st.session_state[StateKey.GLOBAL].project_paths.project_dir
        ):
            DBConnection.set_project_path(project_dir)
            st.session_state[StateKey.GLOBAL] = State(
                project_paths=ProjectFileStructure(project_dir),
                refresh_projects=refresh_projects,
                merged_metrics=MergedMetrics().all(),
                all_tags=Tags().all(),
            )


def has_state():
    return StateKey.GLOBAL in st.session_state


def get_state() -> State:
    if not has_state():
        st.stop()

    return st.session_state[StateKey.GLOBAL]


def refresh(
    *,
    clear_global: bool = False,
    clear_scoped: bool = False,
    clear_memo: bool = False,
    clear_component: bool = False,
    nuke: bool = False,
):
    if nuke:
        st.session_state.clear()
    else:
        if clear_global:
            st.session_state.pop(StateKey.GLOBAL, None)
        if clear_scoped:
            st.session_state.get(StateKey.SCOPED, {}).clear()
        if clear_memo:
            st.session_state.get(StateKey.MEMO, {}).clear()
        if clear_component:
            keys = {key for key in st.session_state.keys() if key not in StateKey.__members__}
            for key in keys:
                st.session_state.pop(key, None)

    st.experimental_rerun()
