from typing import Dict, List, Optional

import streamlit as st
from pandas import Series

from encord_active.app.common.components.tags.tag_creator import scoped_tags
from encord_active.app.common.state import get_state
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import METRIC_SCOPE_TAG_SCOPES, Tag, TagScope
from encord_active.lib.metrics.utils import MetricScope


def target_identifier(identifier: str, scope: TagScope) -> Optional[str]:
    parts = identifier.split("_")
    data, object = parts[:3], parts[3:]

    if scope == TagScope.DATA:
        return "_".join(data)
    elif scope == TagScope.LABEL and object:
        return identifier
    else:
        return None


def update_tags(identifier: str, key: str):
    tags_for_update: List[Tag] = st.session_state[key]

    targeted_tags: Dict[str, List[Tag]] = {}
    for scope in TagScope:
        target_id = target_identifier(identifier, scope)
        if target_id:
            targeted_tags[target_id] = [tag for tag in tags_for_update if tag.scope == scope]

    for id, tags in targeted_tags.items():
        get_state().merged_metrics.at[id, "tags"] = tags
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
        tag_status += get_state().merged_metrics.at[id, "tags"]

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
