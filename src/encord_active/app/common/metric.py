from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st


@dataclass
class MetricData:
    name: str
    path: Path
    meta: Dict[str, Any]
    level: str


@st.cache(allow_output_mutation=True)
def load_metric(metric: MetricData, normalize: bool, *, sorting_key="score") -> pd.DataFrame:
    """
    Load and sort the selected csv file and cache it, so we don't need to perform this
    heavy computation each time the slider in the UI is moved.
    :param metric: The metric to load data from.
    :param normalize: whether to apply normalisation to the scores or not.
    :param sorting_key: key by which to sort dataframe (default: "score")
    :return: a pandas data frame with all the scores.
    """
    df = pd.read_csv(metric.path).sort_values([sorting_key, "identifier"], ascending=True).reset_index()

    if normalize:
        min_val = metric.meta.get("min_value")
        max_val = metric.meta.get("max_value")
        if min_val is None:
            min_val = df["score"].min()
        if max_val is None:
            max_val = df["score"].max()

        diff = max_val - min_val
        if diff == 0:  # Avoid dividing by zero
            diff = 1.0

        df["score"] = (df["score"] - min_val) / diff

    return df
