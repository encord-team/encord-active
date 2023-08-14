import json
import uuid
from typing import List, Optional, Type, Union

import pandas as pd
from pydantic import BaseModel
from sqlalchemy.sql.operators import eq, in_op, or_
from sqlmodel import Session, select

from encord_active.db.models import (
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectTag,
    ProjectTaggedAnnotation,
    ProjectTaggedDataUnit,
)
from encord_active.lib.common.utils import partial_column
from encord_active.lib.db.helpers.tags import GroupedTags, Tag, from_grouped_tags
from encord_active.lib.db.tags import TagScope
from encord_active.lib.metrics.utils import MetricScope
from encord_active.lib.model_predictions.types import PredictionsFilters
from encord_active.lib.project.project_file_structure import ProjectFileStructure
from encord_active.server.dependencies import engine

UNTAGGED_FRAMES_LABEL = "Untagged frames"
UNTAGGED_ANNOTATIONS_LABEL = "Untagged annotations"
NO_CLASS_LABEL = "No class"


class DatetimeRange(BaseModel):
    min: pd.Timestamp
    max: pd.Timestamp


class Range(BaseModel):
    min: float
    max: float


class Filters(BaseModel):
    tags: Optional[GroupedTags] = None
    object_classes: Optional[List[str]] = None
    workflow_stages: Optional[List[str]] = None
    text: dict[str, str] = {}
    categorical: dict[str, List[float]] = {}
    range: dict[str, Range] = {}
    datetime_range: dict[str, DatetimeRange] = {}
    prediction_filters: Optional[PredictionsFilters] = None

    def __hash__(self):
        return hash(json.dumps(self.dict(exclude_defaults=True)))


def apply_filters(df: pd.DataFrame, filters: Filters, pfs: ProjectFileStructure, scope: Optional[MetricScope] = None):
    project_hash = uuid.UUID(pfs.load_project_meta()["project_hash"])
    filtered = df.copy()
    identifier_split = filtered.index.str.split("_", n=3)
    filtered["data_row_id"] = identifier_split.str[0:3].str.join("_")
    filtered["is_label_metric"] = identifier_split.str.len() > 3

    if filters.tags is not None and filters.tags:
        data_tags, label_tags = from_grouped_tags(filters.tags)
        filtered = filter_tags(project_hash, filtered, data_tags, label_tags)

    if filters.object_classes is not None:
        filtered = filter_object_classes(filtered, filters.object_classes, scope)

    if filters.workflow_stages is not None:
        filtered = filter_workflow_stages(filtered, filters.workflow_stages, pfs)

    for column, categorical_filter in filters.categorical.items():
        if column in filtered:
            non_applicable = filtered[pd.isna(filtered[column])]
            filtered = filtered[filtered[column].isin(categorical_filter)]
            filtered = add_non_applicable(filtered, non_applicable)

    for column, range_filter in list(filters.range.items()) + list(filters.datetime_range.items()):
        if column in filtered:
            non_applicable = filtered[pd.isna(filtered[column])]
            filtered = filtered.loc[filtered[column].between(range_filter.min, range_filter.max)]
            filtered = add_non_applicable(filtered, non_applicable)

    for column, text_filter in filters.text.items():
        if column in filtered:
            filtered = filtered[filtered[column].astype(str).str.contains(text_filter)]

    filtered.drop(columns=["data_row_id", "is_label_metric"], inplace=True)

    return filtered


def filter_tags(project_hash, to_filter: pd.DataFrame, data_tags: list[Tag], label_tags: list[Tag]):
    if not (data_tags or label_tags):
        return to_filter

    df = to_filter.copy()

    def _filter_tags_table(
        tags_table: Union[Type[ProjectTaggedDataUnit], Type[ProjectTaggedAnnotation]],
        tags: list[Tag],
        untagged_name: Optional[str],
    ) -> set[str]:
        is_annotations = ProjectTaggedAnnotation == tags_table
        with Session(engine) as sess:
            selected_tag_hashes = sess.exec(
                select(ProjectTag.tag_hash).where(
                    ProjectTag.project_hash == project_hash, in_op(ProjectTag.name, [t.name for t in tags])
                )
            ).all()
            column_selection = [
                ProjectDataMetadata.label_hash,
                ProjectDataUnitMetadata.du_hash,
                ProjectDataUnitMetadata.frame,
            ]
            if is_annotations:
                column_selection.append(ProjectTaggedAnnotation.object_hash)
            stmt = (
                select(*column_selection)  # type: ignore
                .join(
                    tags_table,
                    (tags_table.du_hash == ProjectDataUnitMetadata.du_hash)
                    & (tags_table.frame == ProjectDataUnitMetadata.frame)
                    & (tags_table.project_hash == ProjectDataUnitMetadata.project_hash),
                    isouter=True,
                )
                .join(
                    ProjectDataMetadata,
                    (ProjectDataUnitMetadata.data_hash == ProjectDataMetadata.data_hash)
                    & (ProjectDataUnitMetadata.project_hash == ProjectDataMetadata.project_hash),
                )
                .where(
                    ProjectDataMetadata.project_hash == project_hash,
                )
            )
            if untagged_name in [t.name for t in tags]:
                stmt = stmt.where(
                    or_(eq(tags_table.tag_hash, None), in_op(tags_table.tag_hash, selected_tag_hashes)),
                )
            else:
                stmt = stmt.where(
                    in_op(tags_table.tag_hash, selected_tag_hashes),
                )

            tagged_rows = sess.exec(stmt).all()
            if is_annotations:
                data_identifiers = set(
                    [f"{label_hash}_{du_hash}_{frame:05d}" for label_hash, du_hash, frame, _ in tagged_rows]
                )
                label_identifiers = set(
                    [
                        f"{label_hash}_{du_hash}_{frame:05d}_{annotation_hash}"
                        for label_hash, du_hash, frame, annotation_hash in tagged_rows
                    ]
                )
                return data_identifiers.union(label_identifiers)
            else:
                data_identifiers = set(
                    [f"{label_hash}_{du_hash}_{frame:05d}" for label_hash, du_hash, frame in tagged_rows]
                )
                return data_identifiers

    if data_tags:
        identifiers = _filter_tags_table(ProjectTaggedDataUnit, data_tags, UNTAGGED_FRAMES_LABEL)
        df = df[df.data_row_id.isin(identifiers)]

    if label_tags:
        search_tags = [t for t in label_tags]
        untagged_subset = None
        if UNTAGGED_ANNOTATIONS_LABEL in [t.name for t in search_tags]:
            search_tags = [t for t in search_tags if t.name != UNTAGGED_ANNOTATIONS_LABEL]
            with Session(engine) as sess:
                all_tags = [Tag(name=t, scope=TagScope.LABEL) for t in sess.exec(select(ProjectTag.name)).all()]
            all_tagged_identifiers = _filter_tags_table(ProjectTaggedAnnotation, all_tags, None)
            untagged_subset = df[~df.index.isin(all_tagged_identifiers)]

        identifiers = _filter_tags_table(ProjectTaggedAnnotation, search_tags, UNTAGGED_ANNOTATIONS_LABEL)
        tagged_subset = df[df.index.isin(identifiers)]
        if untagged_subset is not None and not untagged_subset.empty:
            df = pd.concat([tagged_subset, untagged_subset])
        else:
            df = tagged_subset
    return df


def filter_object_classes(to_filter: pd.DataFrame, classes: List[str], scope: Optional[MetricScope] = None):
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
    elif scope == MetricScope.DATA:
        return to_filter[to_filter.index.isin(partial_column(filtered_user_input.index, 3))]
    else:
        return filtered_user_input


def filter_workflow_stages(to_filter: pd.DataFrame, stages: List[str], pfs: ProjectFileStructure):
    label_hashes = to_filter.index.str.split("_", n=1).str[0]
    lr_metadata = json.loads(pfs.label_row_meta.read_text(encoding="utf-8"))
    filtered_label_hashes = {
        lr_hash
        for lr_hash, metadata in lr_metadata.items()
        if metadata.get("workflow_graph_node", dict()).get("title", None) in stages
    }
    return to_filter[label_hashes.isin(filtered_label_hashes)]


def add_non_applicable(df: pd.DataFrame, non_applicable: pd.DataFrame):
    filtered_objects_df = non_applicable[non_applicable.data_row_id.isin(df["data_row_id"])]
    return pd.concat([df, filtered_objects_df])
