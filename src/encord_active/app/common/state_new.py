import inspect
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, List, Optional, TypeVar, Union, overload

import pandas as pd
import streamlit as st

from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tag, Tags
from encord_active.lib.metrics.utils import MetricData
from encord_active.lib.project.project_file_structure import ProjectFileStructure

T = TypeVar("T")
Reducer = Callable[[T], T]

SCOPED_STATES = "scoped_states"
GLOBAL_STATE = "global_state"


def create_key():
    frame = inspect.stack()[2]
    return f"{frame.filename}:{frame.function}:{frame.lineno}"


def use_state(initial: T, key: Optional[str] = None):
    key = key or create_key()
    st.session_state.setdefault(SCOPED_STATES, {}).setdefault(key, initial)

    @overload
    def set_state(arg: T):
        ...

    @overload
    def set_state(arg: Reducer[T]):
        ...

    def set_state(arg: Union[T, Reducer[T]]):
        if callable(arg):
            st.session_state[SCOPED_STATES][key] = arg(st.session_state[SCOPED_STATES][key])
        else:
            st.session_state[SCOPED_STATES][key] = arg

    def get_state() -> T:
        return st.session_state[SCOPED_STATES][key]

    return get_state, set_state


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
    selected_classes: List[str] = field(default_factory=list)
    selected_metric: Optional[MetricData] = None
    page_grid_settings = PageGridSettings()
    normalize_metrics = False

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
