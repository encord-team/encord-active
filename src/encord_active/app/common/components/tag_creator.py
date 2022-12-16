from typing import List, Set

import streamlit as st

import encord_active.app.common.state as state
from encord_active.app.data_quality.common import MetricType
from encord_active.app.db.tags import Tag, Tags, TagScope

SCOPE_EMOJI = {
    TagScope.DATA.value: "🖼️",
    TagScope.LABEL.value: "✏️",
    TagScope.PREDICTION.value: "🤖",
}

METRIC_TYPE_SCOPES = {
    MetricType.DATA_QUALITY: {TagScope.DATA},
    MetricType.LABEL_QUALITY: {TagScope.DATA, TagScope.LABEL},
    MetricType.MODEL_QUALITY: {TagScope.DATA, TagScope.LABEL, TagScope.PREDICTION},
}


def scoped_tags(scopes: Set[TagScope]) -> List[Tag]:
    all_tags: List[Tag] = st.session_state.get(state.ALL_TAGS) or []
    return [tag for tag in all_tags if tag.scope in scopes]


def on_tag_entered(all_tags: List[Tag], name: str, scope: str):
    tag = Tag(f"{SCOPE_EMOJI[scope]} {name}", scope)

    if tag in all_tags:
        st.error("Tag is already in project tags")
        return

    Tags().create_tag(tag)
    all_tags.append(tag)

    st.session_state[state.ALL_TAGS] = all_tags


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
    all_tags = Tags().all()
    if not all_tags:
        return

    tag_counts = st.session_state[state.MERGED_DATAFRAME]["tags"].value_counts()
    all_counts = {name: 0 for name, _ in all_tags}
    for unique_list, count in tag_counts.items():
        for name, *_ in unique_list:
            all_counts[name] = all_counts.get(name, 0) + count

    sorted_tags = sorted(all_counts.items(), key=lambda x: x[0][0].lower())
    st.markdown(
        f"""
    <div class="data-tag-container">
        {' '.join((f'<span class="data-tag">{name} <span class="data-tag data-tag-count">{cnt}</span></span>' for name, cnt in sorted_tags))}
    </div>
    """,
        unsafe_allow_html=True,
    )
