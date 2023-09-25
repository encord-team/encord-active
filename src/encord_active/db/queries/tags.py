import uuid
from typing import Tuple

from sqlmodel import select
from sqlmodel.sql.expression import Select, SelectOfScalar

from encord_active.db.models import (
    ProjectTag,
    ProjectTaggedAnnotation,
    ProjectTaggedDataUnit,
)


def select_frame_data_tags(project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int) -> "SelectOfScalar[ProjectTag]":
    return (
        select(ProjectTag)
        .join(ProjectTaggedDataUnit)
        .where(
            ProjectTaggedDataUnit.project_hash == project_hash,
            ProjectTaggedDataUnit.du_hash == du_hash,
            ProjectTaggedDataUnit.frame == frame,
            ProjectTaggedDataUnit.tag_hash == ProjectTag.tag_hash,
        )
    )


def select_frame_label_tags_distinct(
    project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int
) -> "SelectOfScalar[ProjectTag]":
    return (
        select(ProjectTag)
        .join(ProjectTaggedAnnotation)
        .where(
            ProjectTaggedAnnotation.project_hash == project_hash,
            ProjectTaggedAnnotation.du_hash == du_hash,
            ProjectTaggedAnnotation.frame == frame,
            ProjectTaggedAnnotation.tag_hash == ProjectTag.tag_hash,
        )
        .distinct()
    )


def select_frame_label_tags(
    project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int
) -> "Select[Tuple[ProjectTag, str]]":
    return (
        select(ProjectTag, ProjectTaggedAnnotation.annotation_hash)
        .join(ProjectTaggedAnnotation)
        .where(
            ProjectTaggedAnnotation.project_hash == project_hash,
            ProjectTaggedAnnotation.du_hash == du_hash,
            ProjectTaggedAnnotation.frame == frame,
            ProjectTaggedAnnotation.tag_hash == ProjectTag.tag_hash,
        )
    )


def select_annotation_label_tags(
    project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int, object_hash: str
) -> "SelectOfScalar[ProjectTag]":
    return (
        select(ProjectTag)
        .join(ProjectTaggedAnnotation)
        .where(
            ProjectTaggedAnnotation.project_hash == project_hash,
            ProjectTaggedAnnotation.du_hash == du_hash,
            ProjectTaggedAnnotation.frame == frame,
            ProjectTaggedAnnotation.annotation_hash == object_hash,
            ProjectTaggedAnnotation.tag_hash == ProjectTag.tag_hash,
        )
    )
