import json
from typing import List, Optional

import pandas as pd
from pydantic import BaseModel

from encord_active.lib.db.helpers.tags import Tag
from encord_active.lib.db.tags import TagScope
from encord_active.lib.project.project_file_structure import (
    ProjectFileStructure,
    is_workflow_project,
)

UNTAGED_FRAMES_LABEL = "Untaged frames"
NO_CLASS_LABEL = "No class"


class DatetimeRange(BaseModel):
    start: pd.Timestamp
    end: pd.Timestamp


class Range(BaseModel):
    start: float
    end: float


class Filters(BaseModel):
    tags: Optional[List[Tag]] = None
    object_classes: Optional[List[str]] = None
    workflow_stages: Optional[List[str]] = None
    text: dict[str, str] = {}
    categorical: dict[str, List[float]] = {}
    range: dict[str, Range] = {}
    datetime_range: dict[str, DatetimeRange] = {}

    def __hash__(self):
        return hash(json.dumps(self.dict(exclude_defaults=True)))


def apply_filters(df: pd.DataFrame, filters: Filters, pfs: ProjectFileStructure):
    filtered = df.copy()
    filtered["data_row_id"] = filtered.index.str.split("_", n=3).str[0:3].str.join("_")
    filtered["is_label_metric"] = filtered.index.str.split("_", n=3).str.len() > 3

    filtered = filter_tags(filtered, filters.tags or [])
    if filters.object_classes is not None:
        filtered = filter_object_classes(filtered, filters.object_classes)

    if filters.workflow_stages is not None:
        filtered = filter_workflow_stages(filtered, filters.workflow_stages, pfs)

    for column, categorical_filter in filters.categorical.items():
        non_applicable = filtered[pd.isna(filtered[column])]
        filtered = filtered[filtered[column].isin(categorical_filter)]
        filtered = add_non_applicable(filtered, non_applicable)

    for column, range_filter in list(filters.range.items()) + list(filters.datetime_range.items()):
        non_applicable = filtered[pd.isna(filtered[column])]
        filtered = filtered.loc[filtered[column].between(range_filter.start, range_filter.end)]
        filtered = add_non_applicable(filtered, non_applicable)

    for column, text_filter in filters.text.items():
        filtered = filtered[filtered[column].astype(str).str.contains(text_filter)]

    filtered.drop(columns=["data_row_id", "is_label_metric"], inplace=True)

    return filtered


def filter_tags(to_filter: pd.DataFrame, tags: List[Tag]):
    df = to_filter.copy()
    non_applicable = None
    for tag in tags:
        # Include frames without annotations if the 'no_tag' meta-tag was selected
        if tag.name == UNTAGED_FRAMES_LABEL:
            data_rows = df[~df.is_label_metric]
            filtered_rows = [len(x) == 0 for x in data_rows["tags"]]
            filtered_items = data_rows.loc[filtered_rows]
            df = df[df.data_row_id.isin(filtered_items["data_row_id"])]
            continue

        filtered_rows = [tag in x for x in df["tags"]]
        filtered_items = df.loc[filtered_rows]
        if tag.scope == TagScope.LABEL:
            non_applicable = df[df.index.isin(filtered_items["data_row_id"])]
            df = df[df.index.isin(filtered_items.index)]
        else:
            df = df[df.data_row_id.isin(filtered_items["data_row_id"])]

    if non_applicable is not None and not non_applicable.empty:
        df = add_non_applicable(df, non_applicable)

    return df


def filter_object_classes(to_filter: pd.DataFrame, classes: List[str]):
    # Include all frames that match the user input, excluding frames without annotations
    filtered_user_input = to_filter[to_filter["object_class"].isin(classes)]

    # Include frames without annotations
    if NO_CLASS_LABEL in classes:
        filtered_labels_df = to_filter[to_filter["is_label_metric"]]
        # Select frames whose column 'object_class' is equal to None and don't have logged label metrics
        filtered_no_annotations = to_filter[
            (~to_filter.data_row_id.isin(filtered_labels_df["data_row_id"])) & (pd.isna(to_filter["object_class"]))
        ]
        return pd.concat([filtered_user_input, filtered_no_annotations])
    else:
        return filtered_user_input


def filter_workflow_stages(to_filter: pd.DataFrame, stages: List[str], pfs: ProjectFileStructure):
    # Add 'workflow_stage' column for posterior filter / export actions
    label_hashes = to_filter.index.str.split("_", n=1).str[0]
    lr_metadata = json.loads(pfs.label_row_meta.read_text(encoding="utf-8"))
    lr_to_workflow_stage = {
        lr_hash: metadata.get("workflow_graph_node", dict()).get("title", None)
        for lr_hash, metadata in lr_metadata.items()
    }
    to_filter["workflow_stage"] = label_hashes.map(lr_to_workflow_stage)

    filtered_user_input = to_filter[to_filter["workflow_stage"].isin(stages)]
    return filtered_user_input


def add_non_applicable(df: pd.DataFrame, non_applicable: pd.DataFrame):
    filtered_objects_df = non_applicable[non_applicable.data_row_id.isin(df["data_row_id"])]
    return pd.concat([df, filtered_objects_df])
