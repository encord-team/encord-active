from enum import Enum
from typing import List, NamedTuple, Optional

import streamlit as st
from pandera.typing import DataFrame

from encord_active.app.common.components.tags.individual_tagging import (
    target_identifier,
)
from encord_active.app.common.components.tags.utils import scoped_tags
from encord_active.app.common.state import get_state
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tag, TagScope
from encord_active.lib.metrics.utils import IdentifierSchema


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


def action_bulk_tags(subset: DataFrame[IdentifierSchema], selected_tags: List[Tag], action: TagAction):
    if not selected_tags:
        return

    all_df = get_state().merged_metrics.copy()

    for selected_tag in selected_tags:
        target_ids = [target_identifier(id, selected_tag.scope) for id in subset.identifier.to_list()]
        for _, tags in all_df.loc[target_ids, "tags"].items():
            if action == TagAction.ADD and selected_tag not in tags:
                tags.append(selected_tag)
            elif action == TagAction.REMOVE and selected_tag in tags:
                tags.remove(selected_tag)

    get_state().merged_metrics = all_df
    MergedMetrics().replace_all(all_df)


def bulk_tagging_form(df: DataFrame[IdentifierSchema], is_predictions: bool = False) -> Optional[TaggingFormResult]:
    with st.expander("Bulk Tagging"):
        with st.form("bulk_tagging"):
            select, level_radio, action_radio, button = st.columns([6, 2, 2, 1])
            allowed_tag_scopes = {TagScope.DATA}
            all_rows_have_objects = df[IdentifierSchema.identifier].map(lambda x: len(x.split("_")) > 3).all()
            if all_rows_have_objects and not is_predictions:
                allowed_tag_scopes.add(TagScope.LABEL)
            allowed_tags = scoped_tags(allowed_tag_scopes)

            selected_tags = select.multiselect(
                label="Tags", options=allowed_tags, format_func=lambda x: x[0], label_visibility="collapsed"
            )
            level = level_radio.radio("Level", ["Page", "Range"], horizontal=True)
            action = action_radio.radio("Action", [a.value for a in TagAction], horizontal=True)
            submitted = button.form_submit_button("Submit")

            if not submitted:
                return None

            return TaggingFormResult(submitted, selected_tags, BulkLevel(level), TagAction(action))
