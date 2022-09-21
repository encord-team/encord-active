from enum import Enum
from typing import List, NamedTuple, Optional

import streamlit as st
from pandas import DataFrame

import encord_active.app.common.state as state
from encord_active.app.db.merged_metrics import MergedMetrics


class TagAction(str, Enum):
    ADD = "Add"
    REMOVE = "Remove"


class BulkLevel(str, Enum):
    PAGE = "Page"
    RANGE = "Range"


class TaggingFormResult(NamedTuple):
    submitted: bool
    tags: List[str]
    level: BulkLevel
    action: TagAction


def action_bulk_tags(subset: DataFrame, selected_tags: List[str], action: TagAction):
    if not selected_tags:
        return

    all_df: DataFrame = st.session_state[state.MERGED_DATAFRAME].copy()

    for identifier, tags in all_df.loc[subset.identifier, "tags"].items():
        if action == TagAction.ADD:
            next = list(set(tags + selected_tags))
        elif action == TagAction.REMOVE:
            next = list(set(tag for tag in tags if tag not in selected_tags))
        else:
            raise Exception(f"Action {action} is not supported")

        all_df.at[identifier, "tags"] = next

    st.session_state[state.MERGED_DATAFRAME] = all_df
    MergedMetrics().replace_all(all_df)


def bulk_tagging_form() -> Optional[TaggingFormResult]:
    with st.expander("Bulk Tagging"):
        with st.form("bulk_tagging"):
            select, level_radio, action_radio, button = st.columns([6, 2, 2, 1])
            all_tags = st.session_state.get(state.ALL_TAGS) or []
            selected_tags = select.multiselect(label="Tags", options=all_tags)
            level = level_radio.radio("Level", ["Page", "Range"], horizontal=True)
            action = action_radio.radio("Action", [a.value for a in TagAction], horizontal=True)
            submitted = button.form_submit_button("Submit")

            if not submitted:
                return None

            return TaggingFormResult(submitted, selected_tags, BulkLevel(level), TagAction(action))
