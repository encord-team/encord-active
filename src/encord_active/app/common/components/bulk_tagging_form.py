from enum import Enum
from typing import List, NamedTuple, Optional

import streamlit as st
from pandas import DataFrame

import encord_active.app.common.state as state
from encord_active.lib.db.merged_metrics import MergedMetrics


class TagAction(str, Enum):
    ADD = "Add"
    REMOVE = "Remove"


class BulkLevel(str, Enum):
    PAGE = "Page"
    RANGE = "Range"


class TaggingFormResult(NamedTuple):
    submitted: bool
    tags: List[Tag]
    level: BulkLevel
    action: TagAction


def action_bulk_tags(subset: DataFrame, selected_tags: List[Tag], action: TagAction):
    if not selected_tags:
        return

    all_df: DataFrame = st.session_state[state.MERGED_DATAFRAME].copy()

    for tag in selected_tags:
        target_ids = [target_identifier(id, tag.scope) for id in subset.identifier.to_list()]
        for id, tags in all_df.loc[target_ids, "tags"].items():
            if action == TagAction.ADD:
                next = list(set(tags + [tag]))
            elif action == TagAction.REMOVE:
                next = list(set(tag for tag in tags if tag != tag))
            else:
                raise Exception(f"Action {action} is not supported")

            all_df.at[id, "tags"] = next

    st.session_state[state.MERGED_DATAFRAME] = all_df
    MergedMetrics().replace_all(all_df)


def bulk_tagging_form(metric_type: MetricType) -> Optional[TaggingFormResult]:
    with st.expander("Bulk Tagging"):
        with st.form("bulk_tagging"):
            select, level_radio, action_radio, button = st.columns([6, 2, 2, 1])
            allowed_tags = scoped_tags(METRIC_TYPE_SCOPES[metric_type])
            selected_tags = select.multiselect(
                label="Tags", options=allowed_tags, format_func=lambda x: x[0], label_visibility="collapsed"
            )
            level = level_radio.radio("Level", ["Page", "Range"], horizontal=True)
            action = action_radio.radio("Action", [a.value for a in TagAction], horizontal=True)
            submitted = button.form_submit_button("Submit")

            if not submitted:
                return None

            return TaggingFormResult(submitted, selected_tags, BulkLevel(level), TagAction(action))
