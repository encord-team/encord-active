from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

from encord_active.app.common.state import get_state
from encord_active.app.common.state_hooks import UseState
from encord_active.lib.premium.model import SearchResponse, TextQuery


@dataclass(frozen=True)
class AssistantMode:
    name: str
    fn: Callable[[str, bool, bool], None]


def filter_merged_metrics_by_query_result(result: SearchResponse):
    current_filter = get_state().filtering_state.merged_metrics
    identifiers = [r.identifier for r in result.result_identifiers]
    if result.is_ordered:
        new_filter = current_filter.loc[identifiers]  # type: ignore
    else:
        identifier_set = set(identifiers)
        new_filter = current_filter[current_filter.index.isin(identifier_set)]

    get_state().filtering_state.merged_metrics = new_filter


def query_with_code(query, same_mode: bool, same_query: bool):
    query_result = UseState[str]("")

    if (
        same_query
        and same_mode
        and query_result.value
        and get_state().filtering_state.last_assistant_result is not None
    ):
        st.markdown(
            f"""```python
{query_result.value}
```
"""
        )
        filter_merged_metrics_by_query_result(get_state().filtering_state.last_assistant_result)  # type: ignore
        return

    with st.spinner():
        res = get_state().querier.search_with_code(
            TextQuery(text=query, identifiers=get_state().filtering_state.merged_metrics.index.tolist())
        )

    if res is not None:
        st.markdown(
            f"""```python
{res.snippet}
```
"""
        )

        query_result.set(res.snippet)
        get_state().filtering_state.last_assistant_result = res

        if not res.result_identifiers:
            st.warning("Didn't find any data based on the query. Try rephrasing your query.")
        else:
            filter_merged_metrics_by_query_result(res)
    else:
        st.warning("Couldn't find any data by code. Reverting to semantic searching.")
        query_semantic(query, same_mode=False, same_query=False)


def query_semantic(query: str, same_mode: bool, same_query: bool):
    if same_query and same_mode and (get_state().filtering_state.last_assistant_result is not None):
        filter_merged_metrics_by_query_result(get_state().filtering_state.last_assistant_result)  # type: ignore
        return

    with st.spinner():
        result = get_state().querier.search_semantics(
            TextQuery(text=query, limit=-1, identifiers=get_state().filtering_state.merged_metrics.index.tolist())
        )
    if not result:
        st.warning("Couldn't find any data matching the query. Try rephrasing your query.")
        return

    get_state().filtering_state.last_assistant_result = result
    filter_merged_metrics_by_query_result(result)


assistant_semantic_mode = AssistantMode("Search", query_semantic)
assistant_code_mode = AssistantMode("Code Generation", query_with_code)
ASSISTANT_MODES = [assistant_semantic_mode, assistant_code_mode]


def render_assistant():
    disabled = not get_state().querier.premium_available
    query_col, query_select_col = st.columns([7, 2])

    query = UseState[str]("")

    old_query = query.value
    with query_col:
        query.set(
            st.text_input(
                label="What data would you like to find?",
                placeholder="What would you like to find?",
                help="Disabled in the open source version" if disabled else "Describe what you are looking for",
                disabled=disabled,
            )
        )

    assistant_mode = UseState[Optional[AssistantMode]](assistant_semantic_mode)
    old_mode = assistant_mode.value
    with query_select_col:
        assistant_mode.set(
            st.selectbox("Search options", options=ASSISTANT_MODES, format_func=lambda x: x.name, index=0)
        )

    if not query.value or assistant_mode.value is None:
        return

    same_query = query.value == old_query
    same_mode = assistant_mode.value == old_mode
    assistant_mode.value.fn(query.value, same_mode, same_query)
