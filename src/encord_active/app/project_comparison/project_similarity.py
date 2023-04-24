import stat
import statistics
from pathlib import Path

import numpy as np
import pandas as pd
import pandera as pa
import plotly.express as px
import streamlit as st
from pandera.typing import DataFrame, Series
from scipy.stats import ks_2samp

from encord_active.app.common.state import get_state
from encord_active.app.common.utils import setup_page
from encord_active.app.projects_page import (
    GetProjectsResult,
    get_projects,
    project_list,
)
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.metrics.metadata import fetch_metrics_meta
from encord_active.lib.metrics.utils import MetricScope, load_available_metrics
from encord_active.lib.project.metadata import fetch_project_meta

METRICS_TO_EXCLUDE = [
    "Random Values on Objects",
    "Random Values on Images",
    "Object Annotation Quality",
    "Image Annotation Quality",
    "Image Difficulty",
]


class ProjectSimilaritySchema(pa.SchemaModel):
    metric: Series[str] = pa.Field()
    similarity_score: Series[float] = pa.Field(coerce=True)


def calculate_metric_similarity(array_1: np.ndarray, array_2: np.ndarray) -> float:
    D, _ = ks_2samp(array_1, array_2)
    return 1 - D


def project_similarity():
    def render():
        setup_page()

        project_metas = {
            project: fetch_project_meta(project) for project in project_list(get_state().global_root_folder)
        }

        selected_project = st.selectbox(
            "Select project to compare",
            options=project_metas,
            format_func=(lambda x: project_metas[x]["project_title"]),
        )

        DBConnection.set_project_path(selected_project)
        merged_metrics_2 = MergedMetrics().all()

        DBConnection.set_project_path(get_state().project_paths.project_dir)
        all_metrics = fetch_metrics_meta(get_state().project_paths)

        metric_similarities = {"metric": [], "similarity_score": []}
        for metric in all_metrics.keys():
            if metric in METRICS_TO_EXCLUDE:
                continue

            if metric not in get_state().merged_metrics:
                continue
            values_1 = get_state().merged_metrics[metric].dropna()

            if values_1.shape[0] == 0:
                continue

            if metric not in merged_metrics_2:
                continue
            values_2 = merged_metrics_2[metric].dropna()

            if values_2.shape[0] == 0:
                continue

            metric_similarities["metric"].append(metric)
            metric_similarities["similarity_score"].append(calculate_metric_similarity(values_1, values_2))

        metric_similarities_df = pd.DataFrame(metric_similarities)
        metric_similarities_df = metric_similarities_df.pipe(ProjectSimilaritySchema)

        st.metric(
            "Average Similarity", f"{metric_similarities_df[ProjectSimilaritySchema.similarity_score].mean():.2f}"
        )

        metric_similarities_df.sort_values(by=ProjectSimilaritySchema.similarity_score, inplace=True)

        fig = px.bar(
            metric_similarities_df,
            x=ProjectSimilaritySchema.similarity_score,
            y=ProjectSimilaritySchema.metric,
            orientation="h",
        )

        st.plotly_chart(fig)

    return render
