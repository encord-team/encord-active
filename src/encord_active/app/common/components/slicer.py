from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st


def render_df_slicer(df: pd.DataFrame, selected_metric: Optional[str]):
    if selected_metric not in df:
        return df

    max_val = float(df[selected_metric].max()) + np.finfo(float).eps.item()
    min_val = float(df[selected_metric].min())

    if max_val <= min_val:
        return df

    step = max(0.01, (max_val - min_val) // 100)
    start, end = st.slider("Choose quality", max_value=max_val, min_value=min_val, value=(min_val, max_val), step=step)
    subset = df[df[selected_metric].between(start, end)]

    return subset
