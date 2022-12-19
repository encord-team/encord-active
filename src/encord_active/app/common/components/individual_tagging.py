from typing import Dict, List

import streamlit as st
from pandas import Series

import encord_active.app.common.state as state
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tags


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


def multiselect_tag(row: Series, key_prefix: str, metric_type: MetricType):
    identifier = row["identifier"]

    if not isinstance(identifier, str):
        st.error("Multiple rows with the same identifier were found. Please create a new issue.")
        return

    metric_scopes = METRIC_TYPE_SCOPES[metric_type]

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


def tag_display_with_counts():
    all_tags = Tags().all()
    if not all_tags:
        return

    tag_counts = st.session_state[state.MERGED_DATAFRAME]["tags"].value_counts()
    all_counts = {tag: 0 for tag in all_tags}
    for unique_list, count in tag_counts.items():
        for tag in unique_list:
            all_counts[tag] = all_counts.get(tag, 0) + count

    sorted_tags = sorted(all_counts.items(), key=lambda x: x[0].lower())
    st.markdown(
        f"""
    <div class="data-tag-container">
        {' '.join((f'<span class="data-tag">{tag} <span class="data-tag data-tag-count">{cnt}</span></span>' for tag, cnt in sorted_tags))}
    </div>
    """,
        unsafe_allow_html=True,
    )


def tag_display(tags: List[str]):
    """
    Usage::

        tags = ['list', 'of', 'tags']
        st.markdown(tag_display(all_tags), unsafe_allow_html=True)

    """
    sorted_tags = sorted(tags, key=lambda x: x.lower())
    return f"""
    <div class="data-tag-container">
        {' '.join((f'<span class="data-tag">{tag}</span>' for tag in sorted_tags))}
    </div>"""


def tag_creator():
    with st.sidebar:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### Tags", unsafe_allow_html=True)

        # Show message after on change.
        if "new_tag_message" in st.session_state:
            st.info(st.session_state.new_tag_message)
            del st.session_state.new_tag_message

        def on_tag_entered():
            tag_name = st.session_state.get("new_tag_name_input", "").strip()
            try:
                Tags().create_tag(tag_name)
            except ValueError as e:
                st.session_state.new_tag_message = str(e)
                return

            st.session_state[state.ALL_TAGS].append(tag_name)

        st.text_input(
            "Tag name",
            label_visibility="collapsed",
            on_change=on_tag_entered,
            placeholder="Enter new tag",
            key="new_tag_name_input",
        )
        tag_display_with_counts()
