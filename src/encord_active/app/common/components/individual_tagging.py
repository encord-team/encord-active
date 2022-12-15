from typing import List

import streamlit as st
from pandas import Series

import encord_active.app.common.state as state
from encord_active.app.db.merged_metrics import MergedMetrics


def update_image_tags(identifier: str, key: str):
    tags = st.session_state[key]
    st.session_state[state.MERGED_DATAFRAME].at[identifier, "tags"] = tags
    MergedMetrics().update_tags(identifier, tags)


def multiselect_tag(row: Series, key_prefix: str):
    identifier = row["identifier"]
    tag_status = st.session_state[state.MERGED_DATAFRAME].at[identifier, "tags"]

    if not isinstance(identifier, str):
        st.error("Multiple rows with the same identifier were found. Please create a new issue.")
        return

    key = f"{key_prefix}_multiselect_{identifier}"

    st.multiselect(
        label="Tag image",
        options=st.session_state.get(state.ALL_TAGS) or [],
        format_func=lambda x: x[0],
        default=tag_status if len(tag_status) else None,
        key=key,
        label_visibility="collapsed",
        on_change=update_image_tags,
        args=(identifier, key),
    )
