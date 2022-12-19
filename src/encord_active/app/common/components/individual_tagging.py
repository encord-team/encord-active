from typing import List

import streamlit as st
from pandas import Series

import encord_active.app.common.state as state
from encord_active.lib.db.helpers.tags import count_of_tags
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import Tags


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
        default=tag_status if len(tag_status) else None,
        key=key,
        label_visibility="collapsed",
        on_change=update_image_tags,
        args=(identifier, key),
    )


def tag_display_with_counts():
    all_counts = count_of_tags(st.session_state[state.MERGED_DATAFRAME])

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
