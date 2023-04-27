from functools import partial
from typing import Dict, List, Optional, Set

import streamlit as st
from pandas import Series

from encord_active.app.common.state import get_state
from encord_active.lib.db.helpers.tags import scoped_tags
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tag, TagScope


def target_identifier(identifier: str, scope: TagScope) -> Optional[str]:
    parts = identifier.split("_")
    data = parts[:3]

    if scope == TagScope.DATA:
        return "_".join(data)
    elif scope == TagScope.LABEL:
        return identifier
    else:
        return None


def update_tags(identifier: str, key: str, scopes: Set[TagScope]):
    if key not in st.session_state:
        return
    tags_for_update: List[Tag] = st.session_state[key]

    targeted_tags: Dict[str, List[Tag]] = {}
    for scope in scopes:
        target_id = target_identifier(identifier, scope)
        if target_id:
            targeted_tags[target_id] = [tag for tag in tags_for_update if tag.scope == scope]

    for id, tags in targeted_tags.items():
        tag_arr = get_state().merged_metrics.at[id, "tags"]
        tag_arr.clear()
        tag_arr.extend(tags)
        MergedMetrics().update_tags(id, tags)


def multiselect_tag(row: Series, key_prefix: str, is_predictions=False):
    identifier = row["identifier"]

    if not isinstance(identifier, str):
        st.error("Multiple rows with the same identifier were found. Please create a new issue.")
        return

    _, _, _, *objects = identifier.split("_")
    metric_scopes = {TagScope.DATA}
    if objects and not is_predictions:
        metric_scopes.add(TagScope.LABEL)

    tag_status = []
    merged_metrics = get_state().merged_metrics
    for scope in metric_scopes:
        id = target_identifier(identifier, scope)
        if id in merged_metrics.index:
            tag_status += get_state().merged_metrics.at[id, "tags"]

    key = f"{key_prefix}_multiselect_{identifier}"

    st.multiselect(
        label="Tag image",
        options=scoped_tags(metric_scopes),
        format_func=lambda x: x.name,
        default=tag_status if len(tag_status) else None,
        key=key,
        label_visibility="collapsed",
        on_change=partial(update_tags, scopes=metric_scopes),
        args=(identifier, key),
    )
