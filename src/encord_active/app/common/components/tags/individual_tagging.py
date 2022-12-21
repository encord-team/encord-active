from typing import Dict, List

import streamlit as st
from pandas import Series

import encord_active.app.common.state as state
from encord_active.app.common.components.tags.tag_creator import scoped_tags
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import METRIC_SCOPE_TAG_SCOPES, Tag, TagScope
from encord_active.lib.metrics.load_metrics import MetricScope


def target_identifier(identifier: str, scope: TagScope) -> str:
    if scope == TagScope.DATA:
        return "_".join(identifier.split("_")[:3])
    else:
        return identifier


def update_tags(identifier: str, key: str):
    tags_for_update: List[Tag] = st.session_state[key]

    targeted_tags: Dict[str, List[Tag]] = {}
    for tag in tags_for_update:
        target_id = target_identifier(identifier, tag.scope)
        targeted_tags.setdefault(target_id, []).append(tag)

    for id, tags in targeted_tags.items():
        st.session_state[state.MERGED_DATAFRAME].at[id, "tags"] = tags
        MergedMetrics().update_tags(id, tags)


def multiselect_tag(row: Series, key_prefix: str, metric_type: MetricScope):
    identifier = row["identifier"]

    if not isinstance(identifier, str):
        st.error("Multiple rows with the same identifier were found. Please create a new issue.")
        return

    metric_scopes = METRIC_SCOPE_TAG_SCOPES[metric_type]

    tag_status = []
    for scope in metric_scopes:
        id = target_identifier(identifier, scope)
        tag_status += st.session_state[state.MERGED_DATAFRAME].at[id, "tags"]

    key = f"{key_prefix}_multiselect_{identifier}"

    st.multiselect(
        label="Tag image",
        options=scoped_tags(metric_scopes),
        format_func=lambda x: x.name,
        default=tag_status if len(tag_status) else None,
        key=key,
        label_visibility="collapsed",
        on_change=update_tags,
        args=(identifier, key),
    )
