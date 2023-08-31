import uuid

from sqlmodel import select
from sqlmodel.sql.expression import SelectOfScalar

from encord_active.db.models import (
    ProjectAnnotationAnalytics,
    ProjectDataAnalytics,
    ProjectPredictionAnalytics,
    ProjectPredictionAnalyticsExtra,
)


def select_data_analytics(
    project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int
) -> "SelectOfScalar[ProjectDataAnalytics]":
    return select(ProjectDataAnalytics).where(
        ProjectDataAnalytics.project_hash == project_hash,
        ProjectDataAnalytics.du_hash == du_hash,
        ProjectDataAnalytics.frame == frame,
    )


def select_annotation_analytics(
    project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int, object_hash: str
) -> "SelectOfScalar[ProjectAnnotationAnalytics]":
    return select(ProjectAnnotationAnalytics).where(
        ProjectAnnotationAnalytics.project_hash == project_hash,
        ProjectAnnotationAnalytics.du_hash == du_hash,
        ProjectAnnotationAnalytics.frame == frame,
        ProjectAnnotationAnalytics.object_hash == object_hash,
    )


def select_prediction_analytics_with_extra(project_hash: uuid.UUID, du_hash: uuid.UUID, frame: int, object_hash: str):
    return select(ProjectPredictionAnalytics, ProjectPredictionAnalyticsExtra).where(
        ProjectPredictionAnalytics.project_hash == project_hash,
        ProjectPredictionAnalytics.du_hash == du_hash,
        ProjectPredictionAnalytics.frame == frame,
        ProjectPredictionAnalytics.object_hash == object_hash,
        ProjectPredictionAnalyticsExtra.prediction_hash == ProjectPredictionAnalytics.prediction_hash,
        ProjectPredictionAnalyticsExtra.du_hash == du_hash,
        ProjectPredictionAnalyticsExtra.frame == frame,
        ProjectPredictionAnalyticsExtra.object_hash == object_hash,
    )