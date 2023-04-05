from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd
import streamlit as st

from encord_active.app.common.state import get_state
from encord_active.app.common.state_hooks import UseState


@dataclass(frozen=True)
class AssistantMode:
    name: str
    fn: Callable[[str, bool, bool], None]


def filter_merged_metrics_by_query_result(result: pd.DataFrame, dictate_sorting_order=True):
    if "identifier" in result.columns:
        result = result.set_index("identifier")

    current_filter = get_state().filtering_state.merged_metrics
    if dictate_sorting_order:
        new_filter = current_filter.loc[result[result.index.isin(current_filter.index)].index]  # type: ignore
    else:
        new_filter = current_filter[current_filter.index.isin(result.index)]

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
        filter_merged_metrics_by_query_result(get_state().filtering_state.last_assistant_result, dictate_sorting_order=False)  # type: ignore
        return

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
        make_query_with_embedding(get_state.querier.query_by_embedding)(query, same_mode=False, same_query=same_query)


def make_query_with_embedding(query_fn):
    def query_with_embedding(query: str, same_mode: bool, same_query: bool):
        if same_query and same_mode and (get_state().filtering_state.last_assistant_result is not None):
            filter_merged_metrics_by_query_result(get_state().filtering_state.last_assistant_result)  # type: ignore
            return

        with st.spinner():
            df, _ = query_fn(query, min_similarity=0.65, identifiers=get_state().filtering_state.merged_metrics)

        if df.empty:
            st.warning("Couldn't find any data of that sort. Try rephrasing your query.")
            return

        st.write(f"Found {df.shape[0]} items")

        get_state().filtering_state.last_assistant_result = df
        filter_merged_metrics_by_query_result(df)

    return query_with_embedding


def render_assistant():
    assistant_search_mode = AssistantMode(
        "Data Search", make_query_with_embedding(get_state().querier.query_by_embedding)
    )
    assistant_clip_mode = AssistantMode("CLIP search", make_query_with_embedding(get_state().querier.search_clip))
    assistant_code_mode = AssistantMode("Code Generation", query_with_code)
    ASSISTANT_MODES = [assistant_clip_mode, assistant_search_mode, assistant_code_mode]

    query_col, query_select_col, clear_col = st.columns([7, 2, 2])

    query = UseState[str]("")
    with clear_col:
        st.write("")
        if st.button("Clear Search"):
            get_state().filtering_state.last_assistant_result = None
            query.set("")

    old_query = query.value
    with query_col:
        query.set(
            st.text_input(
                label="What data would you like to find?",
                value=query.value,
                placeholder="What would you like to find?",
            )
        )

    # st.experimental_show(old_query, query.value)

    assistant_mode = UseState[Optional[AssistantMode]](assistant_search_mode)
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
