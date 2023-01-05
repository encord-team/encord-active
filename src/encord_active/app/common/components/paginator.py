import pandas as pd
import streamlit as st


def render_pagination(df: pd.DataFrame, n_cols: int, n_rows: int, sort_key: str):
    n_items = n_cols * n_rows
    col1, col2 = st.columns(spec=[1, 4])

    with col1:
        sorting_order = st.selectbox("Sort samples within selected interval", ["Ascending", "Descending"])

    with col2:
        last = len(df) // n_items + 1
        page_num = st.slider("Page", 1, last) if last > 1 else 1

    low_lim = (page_num - 1) * n_items
    high_lim = page_num * n_items

    sorted_subset = df.sort_values(by=sort_key, ascending=sorting_order == "Ascending")
    paginated_subset = sorted_subset[low_lim:high_lim]
    return paginated_subset
