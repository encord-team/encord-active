import io
import json
from encodings import utf_8
from typing import Dict, List
from zipfile import ZipFile

import numpy as np
import pandas as pd
from tqdm import tqdm

from encord_active.lib.coco.encoder import generate_coco_file
from encord_active.lib.metrics.utils import MetricData, load_metric_dataframe
from encord_active.lib.project import ProjectFileStructure


def balance_dataframe(selected_metrics: List[MetricData], partition_sizes: Dict[str, int], seed: int) -> pd.DataFrame:
    """
    Balances the dataset over the selected metrics and partition sizes.
    Currently, it is done by random sampling.

    Args:
        selected_metrics (List[MetricData]): The metrics to balance over.
        partition_sizes (Dict[str,int]): The dictionary of partition names : partition sizes.
        seed (int): The seed for the random sampling.

    Returns:
        pd.Dataframe: A dataframe with the following columns: sample identifiers, metric values and allocated partition.
    """
    # Collect metric dataframes
    merged_df_list = []
    for i, m in enumerate(selected_metrics):
        df = load_metric_dataframe(m, normalize=False).copy()
        merged_df_list.append(df[["identifier", "score"]].rename(columns={"score": m.name}))

    # Merge all dataframes by identifier
    merged_df = merged_df_list.pop()
    for df_tmp in merged_df_list:
        merged_df = merged_df.merge(df_tmp, on="identifier", how="outer")

    # Randomly sample from each partition and add column to merged_df
    n_samples = len(merged_df)
    selection_df = merged_df.copy()
    merged_df["partition"] = ""
    for partition_name, partition_size in [(k, v) for k, v in partition_sizes.items()][:-1]:
        n_partition = int(np.floor(n_samples * partition_size / 100))
        partition_df = selection_df.sample(n=n_partition, replace=False, random_state=seed)
        # Remove samples from selection_df
        selection_df = selection_df[~selection_df["identifier"].isin(partition_df["identifier"])]
        # Add partition column to merged_df
        merged_df.loc[partition_df.index, "partition"] = partition_name

    # Assign the remaining samples to the last partition
    merged_df.loc[merged_df["partition"] == "", "partition"] = list(partition_sizes.keys())[-1]
    return merged_df


def get_partitions_zip(partition_dict: Dict[str, pd.DataFrame], project_file_structure: ProjectFileStructure) -> bytes:
    """
    Creates a zip file with a COCO json object for each partition.

    Args:
        partition_dict (Dict[str, pd.DataFrame]): A dictionary of partition names : partition dataframes.

    Returns:
        bytes: The zip file as a byte array.
    """
    zip_io = io.BytesIO()
    with ZipFile(zip_io, mode="w") as zf:
        partition_dict.pop("Unassigned", None)
        for partition_name, partition in tqdm(partition_dict.items(), desc="Generating COCO files"):
            coco_json = generate_coco_file(
                partition, project_file_structure.project_dir, project_file_structure.ontology
            )
            with zf.open(partition_name.replace(" ", "_").lower() + ".json", "w") as zip_file:
                writer = utf_8.StreamWriter(zip_file)
                json.dump(coco_json, writer)  # type: ignore
    zip_io.seek(0)

    return zip_io.read()
