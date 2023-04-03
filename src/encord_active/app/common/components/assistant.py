from dataclasses import dataclass
from typing import Callable

import pandas as pd
import streamlit as st

from encord_active.app.common.components.top_padding import top_padding
from encord_active.app.common.state import get_state
from encord_active.app.common.state_hooks import UseState


@dataclass(frozen=True)
class AssistantMode:
    name: str
    fn: Callable[[str, bool], None]


def filter_merged_metrics_by_query_result(result: pd.DataFrame, dictate_sorting_order=True):
    if "identifier" in result.columns:
        result = result.set_index("identifier")

    current_filter = get_state().filtering_state.merged_metrics
    if dictate_sorting_order:
        new_filter = current_filter.loc[result[result.index.isin(current_filter.index)].index]  # type: ignore
    else:
        new_filter = current_filter[current_filter.index.isin(result.index)]

    get_state().filtering_state.merged_metrics = new_filter


def query_with_code(query, same_mode: bool):
    query_state = UseState[str]("")
    query_result = UseState[str]("")

    if (
        query_state.value == query
        and query_result.value
        and same_mode
        and get_state().filtering_state.last_assistant_result is not None
    ):
        st.markdown(
            f"""```python
{query_result.value}
```
"""
        )
        filter_merged_metrics_by_query_result(get_state().filtering_state.last_assistant_result, dictate_sorting_order=False)  # type: ignore
        return

    query_state.set(query)
    with st.spinner():
        res = get_state().querier.query_with_code(query, identifiers=get_state().filtering_state.merged_metrics)

    if res is not None:
        merged_metric_subset, executed_code = res
        st.markdown(
            f"""```python
{executed_code}
```
"""
        )

        query_result.set(executed_code)
        get_state().filtering_state.last_assistant_result = merged_metric_subset

        if merged_metric_subset.empty:
            st.warning("Didn't find any data based on the query. Try rephrasing your query.")
        else:
            st.write(f"Found {merged_metric_subset.shape[0]} items")
            filter_merged_metrics_by_query_result(merged_metric_subset, dictate_sorting_order=False)
    else:
        st.warning("Couldn't find any data by code. Reverting to searching.")
        query_with_embedding(query, same_mode=False)


def query_with_embedding(query: str, same_mode: bool):
    query_state = UseState[str]("")

    if (query_state.value == query) and (get_state().filtering_state.last_assistant_result is not None) and same_mode:
        filter_merged_metrics_by_query_result(get_state().filtering_state.last_assistant_result)  # type: ignore
        return

    query_state.set(query)
    with st.spinner():
        df, _ = get_state().querier.query_by_embedding(
            query, min_similarity=0.65, identifiers=get_state().filtering_state.merged_metrics
        )

    if df.empty:
        st.warning("Couldn't find any data of that sort. Try rephrasing your query.")
        return

    st.write(f"Found {df.shape[0]} items")

    get_state().filtering_state.last_assistant_result = df
    filter_merged_metrics_by_query_result(df)


assistant_search_mode = AssistantMode("Data Search", query_with_embedding)
assistant_code_mode = AssistantMode("Code Generation", query_with_code)
ASSISTANT_MODES = [assistant_search_mode, assistant_code_mode]


def render_assistant():
    query_col, query_select_col, clear_col = st.columns([7, 2, 2])

    query = UseState[str]("")
    with clear_col:
        st.write("")
        if st.button("Clear Search"):
            get_state().filtering_state.last_assistant_result = None
            query.set("")

    with query_col:
        query.set(
            st.text_input(
                label="What data would you like to find?",
                value=query.value,
                placeholder="What would you like to find?",
            )
        )

    assistant_mode = UseState[AssistantMode](assistant_search_mode)
    old_mode = assistant_mode.value
    with query_select_col:
        assistant_mode.set(
            st.selectbox("Search options", options=ASSISTANT_MODES, format_func=lambda x: x.name, index=0)
        )

    if not query.value or not assistant_mode:
        return

    same_mode = assistant_mode.value == old_mode
    assistant_mode.value.fn(query.value, same_mode)
