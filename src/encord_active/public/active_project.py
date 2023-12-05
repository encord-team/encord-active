from dataclasses import dataclass
from pathlib import Path
from typing import Union
from uuid import UUID, uuid4

import pandas as pd
from encord.objects import OntologyStructure
from sqlalchemy import Integer
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from encord_active.db.models import (
    Project,
    ProjectDataAnalytics,
    ProjectPrediction,
    ProjectPredictionAnalytics,
    ProjectTag,
    ProjectTaggedDataUnit,
    get_engine,
)

_P = Project
_T = ProjectTag


@dataclass
class DataUnitItem:
    du_hash: UUID
    frame: int


def get_active_engine(path_to_db: Union[str, Path]) -> Engine:
    path = Path(path_to_db) if isinstance(path_to_db, str) else path_to_db
    return get_engine(path, use_alembic=False)


class ActiveProject:
    def __init__(self, engine: Engine, project_name: str):
        self._engine = engine
        self._project_name = project_name

        with Session(self._engine) as sess:
            res = sess.exec(
                select(_P.project_hash, _P.project_ontology).where(_P.project_name == project_name).limit(1)
            ).first()

            if res is None:
                raise ValueError(f"Couldn't find project with name `{project_name}` in the DB.")

            self.project_hash, ontology = res
            project_tuples = sess.exec(select(_T.name, _T.tag_hash).where(_T.project_hash == self.project_hash)).all()

            # Assuming that there's just one prediction model
            # FIXME: With multiple sets of model predictions, we should select the right UUID here
            self._model_hash = sess.exec(
                select(ProjectPrediction.prediction_hash)
                .where(ProjectPrediction.project_hash == self.project_hash)
                .limit(1)
            ).first()

        self._existing_tags = dict(project_tuples)
        self.ontology = OntologyStructure.from_dict(ontology)  # type: ignore

    # For backward compatibility
    def get_ontology(self) -> OntologyStructure:
        return self.ontology

    def get_prediction_metrics(self) -> pd.DataFrame:
        if self._model_hash is None:
            raise ValueError(f"Project with name {self._project_name} does not have any model predictions")

        with Session(self._engine) as sess:
            P = ProjectPredictionAnalytics
            stmt = select(  # type: ignore
                P.du_hash,
                P.feature_hash,
                P.metric_area,
                P.metric_area_relative,
                P.metric_aspect_ratio,
                P.metric_brightness,
                P.metric_contrast,
                P.metric_sharpness,
                P.metric_red,
                P.metric_green,
                P.metric_blue,
                P.metric_label_border_closeness,
                P.metric_label_confidence,
                (P.feature_hash == P.match_feature_hash).cast(Integer).label("true_positive"),  # type: ignore
            ).where(P.project_hash == self.project_hash, P.prediction_hash == self._model_hash)

            df = pd.DataFrame(sess.exec(stmt).all())
            df.columns = list(sess.exec(stmt).keys())

        return df

    def get_images_metrics(self) -> pd.DataFrame:
        with Session(self._engine) as sess:
            image_metrics = sess.exec(
                select(ProjectDataAnalytics).where(ProjectDataAnalytics.project_hash == self.project_hash)
            )
            df = pd.DataFrame([i.dict() for i in image_metrics])

        return df

    def get_or_add_tag(self, tag: str) -> UUID:
        with Session(self._engine) as sess:
            if tag in self._existing_tags:
                return self._existing_tags[tag]

            tag_hash = uuid4()
            new_tag = ProjectTag(
                tag_hash=tag_hash,
                name=tag,
                project_hash=self.project_hash,
                description="",
            )
            sess.add(new_tag)
            self._existing_tags[tag] = new_tag.tag_hash
            sess.commit()
        return tag_hash

    def add_tag_to_data_units(
        self,
        tag_hash: UUID,
        du_items: list[DataUnitItem],
    ) -> None:
        with Session(self._engine) as sess:
            for du_item in du_items:
                sess.add(
                    ProjectTaggedDataUnit(
                        project_hash=self.project_hash,
                        du_hash=du_item.du_hash,
                        frame=du_item.frame,
                        tag_hash=tag_hash,
                    )
                )
            sess.commit()
