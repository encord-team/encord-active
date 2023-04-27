import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import ks_2samp

from encord_active.app.common.state import get_state
from encord_active.app.common.utils import setup_page
from encord_active.app.projects_page import project_list
from encord_active.lib.charts.project_similarity import (
    ProjectSimilaritySchema,
    plot_project_similarity_metric_wise,
    render_2d_metric_similarity_plot,
)
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.metrics.metadata import fetch_metrics_meta
from encord_active.lib.project import ProjectFileStructure
from encord_active.lib.project.metadata import fetch_project_meta

# The following metrics does not give any information comparable to another project
METRICS_TO_EXCLUDE = [
    "Random Values on Objects",
    "Random Values on Images",
    "Image Difficulty",
]


def calculate_metric_similarity(array_1: np.ndarray, array_2: np.ndarray) -> float:
    ks_score, _ = ks_2samp(array_1, array_2)
    return 1 - ks_score


def render_2d_metric_similarity_container(
    all_metrics: dict, merged_metrics_1: pd.DataFrame, merged_metrics_2: pd.DataFrame, project_name_2: str
):
    metrics_filtered = {}
    for metric_name in all_metrics.keys():
        if metric_name in merged_metrics_1 and merged_metrics_1[metric_name].dropna().shape[0] > 0:
            metrics_filtered[metric_name] = all_metrics[metric_name]

    if len(metrics_filtered) <= 1:
        st.write("At least two metrics should be generated for 2D metric similarity plot")
        return

    metric_selection_col_1, metric_selection_col_2 = st.columns(2)
    metric_name_1 = metric_selection_col_1.selectbox(
        "Select the first metric",
        options=metrics_filtered,
        index=0,
        format_func=(lambda x: metrics_filtered[x]["title"]),
        key=f"project_comparison_metric_selection_1_{get_state().project_paths.project_dir.name}_{project_name_2}",
    )

    # Second selectbox should be populated according to the selected metric in the first one
    # due to different metric types. E.g., 'Area' is not compatible with 'object aspect ratio'
    options_for_second_metric = {}

    for metric_name in metrics_filtered.keys():
        if metric_name == metric_name_1:
            continue
        metric_df_1 = merged_metrics_1[metric_name_1].dropna().to_frame()
        metric_df_2 = merged_metrics_1[metric_name].dropna().to_frame()

        merged_metric = metric_df_1.join(metric_df_2, how="inner")
        if merged_metric.shape[0] > 0:
            options_for_second_metric[metric_name] = metrics_filtered[metric_name]

    if len(options_for_second_metric) == 0:
        st.write("There is no compatible metric")
        return

    metric_name_2 = metric_selection_col_2.selectbox(
        "Select the second metric",
        options=options_for_second_metric,
        format_func=(lambda x: options_for_second_metric[x]["title"]),
        key=f"project_comparison_metric_selection_2_{get_state().project_paths.project_dir.name}_{project_name_2}",
    )

    if metric_name_1 == metric_name_2:
        st.warning("Please choose a different metric")
        return

    project_values_1 = merged_metrics_1[[metric_name_1, metric_name_2]].copy().dropna()
    project_name_1 = get_state().project_paths.project_dir.name
    project_values_1["project"] = project_name_1

    # Check if the selected metrics are also available in the second project
    if metric_name_1 not in merged_metrics_2 or merged_metrics_2[metric_name_1].dropna().shape[0] == 0:
        st.write(f"Metric **{metric_name_1}** is not available for the project `{project_name_2}`.")
        return

    if metric_name_2 not in merged_metrics_2 or merged_metrics_2[metric_name_2].dropna().shape[0] == 0:
        st.write(f"Metric **{metric_name_2}** is not available for the project `{project_name_2}`.")
        return

    project_values_2 = merged_metrics_2[[metric_name_1, metric_name_2]].copy().dropna()
    project_values_2["project"] = project_name_2

    # TODO change append to concat
    project_values = project_values_1.append(project_values_2, ignore_index=True)

    fig = render_2d_metric_similarity_plot(project_values, metric_name_1, metric_name_2, project_name_1, project_name_2)
    st.plotly_chart(fig, use_container_width=True)


def project_similarity():
    def render():
        setup_page()

        project_metas = {project: fetch_project_meta(project) for project in project_list(get_state().target_path)}
        project_metas = {k: v for k, v in sorted(project_metas.items(), key=lambda item: item[0])}

        selected_project = st.selectbox(
            "Select project to compare",
            options=project_metas,
            index=0,
            format_func=(lambda x: project_metas[x]["project_title"]),
            key=f"project_similarity_{get_state().project_paths.project_dir.name}",
        )

        # TODO this is a hacky way to get MergedMetrics of another project, it should be fixed later
        DBConnection.set_project_file_structure(ProjectFileStructure(selected_project))
        merged_metrics_2 = MergedMetrics().all()
        project_name_2 = project_metas[selected_project]["project_title"]
        DBConnection.set_project_file_structure(ProjectFileStructure(get_state().project_paths.project_dir))

        all_metrics = fetch_metrics_meta(get_state().project_paths)

        metric_similarities = {"metric": [], "similarity_score": []}
        for metric in all_metrics.keys():
            if metric in METRICS_TO_EXCLUDE:
                continue

            if (metric not in get_state().merged_metrics) or (
                get_state().merged_metrics[metric].dropna().shape[0] == 0
            ):
                continue
            values_1 = get_state().merged_metrics[metric].dropna()

            if (metric not in merged_metrics_2) or (merged_metrics_2[metric].dropna().shape[0] == 0):
                continue
            values_2 = merged_metrics_2[metric].dropna()

            metric_similarities["metric"].append(metric)
            metric_similarities["similarity_score"].append(calculate_metric_similarity(values_1, values_2))

        metric_similarities_df = pd.DataFrame(metric_similarities).pipe(ProjectSimilaritySchema)

        metric_wise_similarity_col, metrics_scatter_col = st.columns(2)

        figure = plot_project_similarity_metric_wise(metric_similarities_df)
        metric_wise_similarity_col.metric(
            "Average Similarity", f"{metric_similarities_df[ProjectSimilaritySchema.similarity_score].mean():.2f}"
        )
        metric_wise_similarity_col.plotly_chart(figure, use_container_width=True)

        # 2D metrics view
        with metrics_scatter_col:
            render_2d_metric_similarity_container(
                all_metrics, get_state().merged_metrics, merged_metrics_2, project_name_2
            )

    return render
