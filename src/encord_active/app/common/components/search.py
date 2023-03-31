import streamlit as st

from encord_active.app.common.state import get_state


def query_with_code(query):
    with st.spinner():
        res = get_state().querier.query_with_code(query)

    if res is not None:
        merged_metric_subset, executed_code = res
        st.markdown(
            f"""```python
{executed_code}
```
"""
        )
        if merged_metric_subset.empty:
            st.warning("Didn't find any data based on the query. Try rephrasing your query.")
        else:
            st.write(f"Found {merged_metric_subset.shape[0]} items")
            get_state().filtering_state.merged_metrics = merged_metric_subset
    else:
        st.warning("Couldn't find any data of that sort. Try rephrasing your query.")


def query_with_embedding(query: str):
    with st.spinner():
        df, similarities = get_state().querier.query_by_embedding(query, min_similarity=0.73, num_results=24)

    if df.empty:
        st.warning("Couldn't find any data of that sort. Try rephrasing your query.")
        return

    st.write(f"Found {df.shape[0]} items")
    get_state().filtering_state.merged_metrics = df


def render_search():
    query_col, query_select_col = st.columns([5, 2])
    with query_col:
        query = st.text_input(label="What data would you like to find?")

    with query_select_col:
        query_with = st.selectbox("Search options", options=["code snippets", "natural language"], index=0)

    if not query:
        return

    (query_with_code if query_with == "code snippets" else query_with_embedding)(query)
