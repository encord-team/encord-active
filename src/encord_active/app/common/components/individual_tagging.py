from typing import List

import streamlit as st
from pandas import Series

import encord_active.app.common.state as state
from encord_active.app.db.merged_metrics import MergedMetrics
from encord_active.app.db.tags import Tags


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

        all_tags = Tags().all()
        st.session_state[state.ALL_TAGS] = all_tags

        def on_tag_entered():
            tag_name = st.session_state.get("new_tag_name_input", "").strip()
            if not tag_name.strip():
                msg = "Cannot add empty tag."
                st.session_state.new_tag_message = msg
                return

            st.session_state.new_tag_name_input = ""

            if tag_name in all_tags:
                st.session_state.new_tag_message = "Tag is already in project tags"
                return

            all_tags.append(tag_name)
            Tags().create_tag(tag_name)

            st.session_state[state.ALL_TAGS] = all_tags

        st.text_input(
            "Tag name",
            label_visibility="collapsed",
            on_change=on_tag_entered,
            placeholder="Enter new tag",
            key="new_tag_name_input",
        )
        tag_display_with_counts()
