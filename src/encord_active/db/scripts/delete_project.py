import uuid
from typing import List, Type, Union

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from encord_active.db.models import (
    Project,
    ProjectAnnotationAnalytics,
    ProjectAnnotationAnalyticsExtra,
    ProjectAnnotationAnalyticsReduced,
    ProjectDataAnalytics,
    ProjectDataAnalyticsExtra,
    ProjectDataAnalyticsReduced,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectEmbeddingReduction,
    ProjectPrediction,
    ProjectPredictionAnalytics,
    ProjectPredictionAnalyticsFalseNegatives,
    ProjectPredictionAnalyticsReduced,
    ProjectTag,
    ProjectTaggedAnnotation,
    ProjectTaggedDataUnit,
)


def delete_project_from_db(engine: Engine, project_hash: uuid.UUID, error_on_missing: bool = True) -> None:
    # Attempt to rely on cascade delete
    with Session(engine) as sess:
        old_project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if old_project is None:
            if error_on_missing:
                raise ValueError("BUG: Could not find old project")
            else:
                return
        sess.delete(old_project)
        sess.commit()

    # Fallback (sqlite can fail to run cascade deletion)
    with Session(engine) as sess:
        prediction_hashes = sess.exec(
            select(ProjectPrediction.prediction_hash).where(ProjectPrediction.project_hash == project_hash)
        ).fetchall()
        project_types: List[
            Type[
                Union[
                    ProjectAnnotationAnalytics,
                    ProjectAnnotationAnalyticsExtra,
                    ProjectAnnotationAnalyticsReduced,
                    ProjectDataAnalytics,
                    ProjectDataAnalyticsExtra,
                    ProjectDataAnalyticsReduced,
                    ProjectDataMetadata,
                    ProjectDataUnitMetadata,
                    ProjectEmbeddingReduction,
                    ProjectTag,
                    ProjectTaggedDataUnit,
                    ProjectTaggedAnnotation,
                ]
            ]
        ] = [
            ProjectAnnotationAnalytics,
            ProjectAnnotationAnalyticsExtra,
            ProjectAnnotationAnalyticsReduced,
            ProjectDataAnalytics,
            ProjectDataAnalyticsExtra,
            ProjectDataAnalyticsReduced,
            ProjectDataMetadata,
            ProjectDataUnitMetadata,
            ProjectEmbeddingReduction,
            ProjectTag,
            ProjectTaggedDataUnit,
            ProjectTaggedAnnotation,
        ]
        prediction_types: List[
            Type[
                Union[
                    ProjectPrediction,
                    ProjectPredictionAnalytics,
                    ProjectPredictionAnalyticsReduced,
                    ProjectPredictionAnalyticsFalseNegatives,
                ]
            ]
        ] = [
            ProjectPrediction,
            ProjectPredictionAnalytics,
            # ProjectPredictionAnalyticsExtra, FIXME: add when this column is added
            ProjectPredictionAnalyticsReduced,
            ProjectPredictionAnalyticsFalseNegatives,
        ]
        for project_ty in project_types:
            values = sess.exec(select(project_ty).where(project_ty.project_hash == project_hash)).fetchall()
            for val in values:
                sess.delete(val)
        for prediction_ty in prediction_types:
            for prediction_hash in prediction_hashes:
                values = sess.exec(
                    select(prediction_ty).where(prediction_ty.prediction_hash == prediction_hash)
                ).fetchall()
                for val in values:
                    sess.delete(val)

        sess.commit()
