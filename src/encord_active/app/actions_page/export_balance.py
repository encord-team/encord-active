from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import streamlit as st

from encord_active.app.common.state import get_state
from encord_active.app.common.state_hooks import use_state
from encord_active.app.common.utils import set_page_config, setup_page
from encord_active.lib.charts.partition_histogram import get_partition_histogram
from encord_active.lib.dataset.balance import balance_dataframe, get_partitions_zip
from encord_active.lib.metrics.utils import (
    MetricData,
    MetricScope,
    load_available_metrics,
)


def metrics_panel() -> Tuple[List[MetricData], int]:
    """
    Panel for selecting the metrics to balance over.

    Returns:
        selected_metrics (List[MetricData]): The metrics to balance over.
        seed (int): The seed for the random sampling.
    """
    # TODO - add label metrics
    metrics = load_available_metrics(get_state().project_paths.metrics, MetricScope.DATA_QUALITY)
    metric_names = [metric.name for metric in metrics]

    col1, col2 = st.columns([6, 1])
    with col1:
        selected_metric_names = st.multiselect(
            label="Filter by metric",
            options=metric_names,
            key="balance_metrics",
        )
    seed = col2.number_input("Seed", value=42, step=1, key="seed")

    if not selected_metric_names:
        selected_metric_names = metric_names
    selected_metrics = [metric for metric in metrics if metric.name in selected_metric_names]
    return selected_metrics, int(seed)


def partitions_panel() -> Dict[str, int]:
    """
    Panel for setting the partition sizes.

    Returns:
        A dictionary with the partition names as keys and the partition sizes as values.
    """
    get_partitions_number, set_partitions_number = use_state(1)

    def add_partition():
        set_partitions_number(lambda prev: prev + 1)

    def remove_partition():
        set_partitions_number(lambda prev: prev - 1)

    partition_sizes = {}
    for i in range(get_partitions_number()):
        partition_columns = st.columns((4, 12, 1))
        partition_name = partition_columns[0].text_input(
            f"Name of partition {i + 1}", key=f"name_partition_{i + 1}", value=f"Partition {i + 1}"
        )
        partition_sizes[partition_name] = partition_columns[1].slider(
            f"Data percentage in partition {i + 1}",
            key=f"size_partition_{i + 1}",
            min_value=1,
            max_value=100,
            value=100 // get_partitions_number(),
            step=1,
        )
        if i > 0:
            partition_columns[2].button("❌", key=f"delete_partition_{i + 1}", on_click=remove_partition)
    st.button("➕ Add partition", on_click=add_partition)

    if sum(partition_sizes.values()) != 100:
        st.warning(
            f"The sum of the partition sizes is not 100%. "
            f"{100-sum(partition_sizes.values())}% of samples will not be assigned to a partition."
        )

        partition_sizes["Unassigned"] = 100 - sum(partition_sizes.values())

    return partition_sizes


def export_balance():
    setup_page()
    st.header("Balance & Export")
    st.write(
        "Here you can create balanced partitions of your dataset over a set of metrics and export them as a CSV file."
    )

    selected_metrics, seed = metrics_panel()
    partition_sizes = partitions_panel()

    with st.spinner("Balancing dataset..."):
        balanced_df = balance_dataframe(selected_metrics=selected_metrics, partition_sizes=partition_sizes, seed=seed)

    # Resulting partition sizes
    with st.expander("Resulting partition sizes"):
        st.warning("Due to rounding errors, the resulting partition sizes might not be exactly as specified. ")
        cols = st.columns(len(partition_sizes))
        partition_dict: Dict[str, pd.DataFrame] = {}
        for col, (partition_name, _) in zip(cols, partition_sizes.items()):
            partition = balanced_df[balanced_df["partition"] == partition_name]
            partition_dict[partition_name] = partition
            n_partition_df = partition.shape[0]
            col.write(
                f"{partition_name} ({round(100*n_partition_df/balanced_df.shape[0], 4)}%): {n_partition_df} samples"
            )

    action_columns = st.columns((3, 3, 8))
    is_pressed = action_columns[0].button(
        "🌀 Generate COCO file",
        help="Generate COCO file with filtered data",
    )

    with st.spinner("Generating COCO files"):
        partitions_zip_file = get_partitions_zip(partition_dict, get_state().project_paths) if is_pressed else ""

    action_columns[1].download_button(
        "⬇ Download filtered data",
        partitions_zip_file,
        file_name=f"encord-active-coco-partitions-{datetime.now().strftime('%Y_%m_%d %H_%M_%S')}.zip",
        disabled=not is_pressed,
        help="Ensure you have generated an updated COCO file before downloading",
    )

    st.subheader("View data distributions")
    # Plot distribution of partitions for each metric
    for m in selected_metrics:
        with st.expander(f"{m.name} - Partition distribution"):
            chart = get_partition_histogram(balanced_df, m.name)
            st.altair_chart(chart, use_container_width=True)


if __name__ == "__main__":
    set_page_config()
    export_balance()
