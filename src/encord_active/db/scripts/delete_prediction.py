import uuid
from typing import List, Type, Union

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from encord_active.db.models import (
    ProjectPrediction,
    ProjectPredictionAnalytics,
    ProjectPredictionAnalyticsExtra,
    ProjectPredictionAnalyticsFalseNegatives,
    ProjectPredictionAnalyticsReduced,
)


def delete_prediction_from_db(
    engine: Engine, project_hash: uuid.UUID, prediction_hash: uuid.UUID, error_on_missing: bool = True
) -> None:
    # Attempt to rely on cascade delete
    with Session(engine) as sess:
        old_prediction = sess.exec(
            select(ProjectPrediction).where(
                ProjectPrediction.project_hash == project_hash,
                ProjectPrediction.prediction_hash == prediction_hash,
            )
        ).first()
        if old_prediction is None:
            if error_on_missing:
                raise ValueError("BUG: Could not find old project")
            else:
                return
        sess.delete(old_prediction)
        sess.commit()

    # Fallback (sqlite can fail to run cascade deletion)
    with Session(engine) as sess:
        prediction_types: List[
            Type[
                Union[
                    ProjectPrediction,
                    ProjectPredictionAnalytics,
                    ProjectPredictionAnalyticsExtra,
                    ProjectPredictionAnalyticsReduced,
                    ProjectPredictionAnalyticsFalseNegatives,
                ]
            ]
        ] = [
            ProjectPrediction,
            ProjectPredictionAnalytics,
            ProjectPredictionAnalyticsExtra,
            ProjectPredictionAnalyticsReduced,
            ProjectPredictionAnalyticsFalseNegatives,
        ]
        for prediction_ty in prediction_types:
            values = sess.exec(select(prediction_ty).where(prediction_ty.prediction_hash == prediction_hash)).fetchall()
            for val in values:
                sess.delete(val)

        sess.commit()
