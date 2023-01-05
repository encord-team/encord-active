from typing import List, Set

import streamlit as st

from encord_active.app.common.state import get_state
from encord_active.lib.db.helpers.tags import count_of_tags
from encord_active.lib.db.tags import SCOPE_EMOJI, Tag, Tags, TagScope


def scoped_tags(scopes: Set[TagScope]) -> List[Tag]:
    all_tags: List[Tag] = get_state().all_tags or []
    return [tag for tag in all_tags if tag.scope in scopes]


def on_tag_entered(all_tags: List[Tag], name: str, scope: TagScope):
    tag = Tag(f"{SCOPE_EMOJI[scope]} {name}", scope)

    if tag in all_tags:
        st.error("Tag is already in project tags")
        return

    Tags().create_tag(tag)
    all_tags.append(tag)

    get_state().all_tags = all_tags


def tag_creator():
    with st.sidebar:
        st.markdown("<hr/>", unsafe_allow_html=True)
        st.markdown("### Tags", unsafe_allow_html=True)

        # Show message after on change.
        if "new_tag_message" in st.session_state:
            st.info(st.session_state.new_tag_message)
            del st.session_state.new_tag_message

        all_tags = Tags().all()
        get_state().all_tags = all_tags

        with st.form("tag_creation_form", clear_on_submit=False):
            left, right = st.columns([2, 1])
            name = st.text_input(
                "Tag name",
                label_visibility="collapsed",
                placeholder="Enter new tag",
            ).strip()
            scope = left.radio(
                "Scope",
                [a.value for a in TagScope],
                label_visibility="collapsed",
                format_func=lambda x: f"{SCOPE_EMOJI[x]} {x}",
            )
            submitted = right.form_submit_button("Submit")

            if submitted and scope and name:
                on_tag_entered(all_tags, name, scope)

        tag_display_with_counts()


def tag_display_with_counts():
    all_counts = count_of_tags(get_state().merged_metrics)

    sorted_tags = sorted(all_counts.items(), key=lambda x: x[0].lower())
    st.markdown(
        f"""
    <div class="data-tag-container">
        {' '.join((f'<span class="data-tag">{name} <span class="data-tag data-tag-count">{cnt}</span></span>' for name, cnt in sorted_tags))}
    </div>
    """,
        unsafe_allow_html=True,
    )
