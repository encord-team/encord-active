from dataclasses import dataclass
from typing import Any
from uuid import UUID, uuid4

import pandas as pd
from encord.objects import OntologyStructure
from sqlalchemy import MetaData, Table, create_engine, select, text
from sqlalchemy.engine import Engine
from sqlalchemy.sql import Select
from sqlmodel import Session
from sqlmodel import select as sqlmodel_select

from encord_active.db.models import ProjectTag, ProjectTaggedDataUnit


@dataclass
class DataUnitItem:
    du_hash: str
    frame: int


def get_active_engine(path_to_db: str) -> Engine:
    return create_engine(f"sqlite:///{path_to_db}")


class ActiveProject:
    def __init__(self, engine: Engine, project_name: str):
        self._engine = engine
        self._metadata = MetaData(bind=self._engine)
        self._project_name = project_name

        active_project = Table("active_project", self._metadata, autoload_with=self._engine)
        stmt = select(active_project.c.project_hash).where(active_project.c.project_name == f"{self._project_name}")

        with self._engine.connect() as connection:
            result = connection.execute(stmt).fetchone()

            if result is not None:
                self._project_hash = result[0]
            else:
                self._project_hash = None

        with Session(engine) as sess:
            self._existing_tags = {
                tag.name: tag.tag_hash
                for tag in sess.exec(
                    sqlmodel_select(ProjectTag).where(ProjectTag.project_hash == self._project_hash)
                ).all()
            }
            sess.commit()

    def _execute_statement(self, stmt: Select) -> Any:
        with self._engine.connect() as connection:
            result = connection.execute(stmt).fetchone()

            if result is not None:
                return result[0]
            else:
                return None

    def get_ontology(self) -> OntologyStructure:
        active_project = Table("active_project", self._metadata, autoload_with=self._engine)

        stmt = select(active_project.c.project_ontology).where(active_project.c.project_hash == f"{self._project_hash}")

        return OntologyStructure.from_dict(self._execute_statement(stmt))

    def get_prediction_metrics(self) -> pd.DataFrame:
        active_project_prediction = Table("active_project_prediction", self._metadata, autoload_with=self._engine)
        stmt = select(active_project_prediction.c.prediction_hash).where(
            active_project_prediction.c.project_hash == f"{self._project_hash}"
        )

        prediction_hash = self._execute_statement(stmt)

        active_project_prediction_analytics = Table(
            "active_project_prediction_analytics", self._metadata, autoload_with=self._engine
        )

        stmt = select(
            [
                active_project_prediction_analytics.c.du_hash,
                active_project_prediction_analytics.c.feature_hash,
                active_project_prediction_analytics.c.metric_area,
                active_project_prediction_analytics.c.metric_area_relative,
                active_project_prediction_analytics.c.metric_aspect_ratio,
                active_project_prediction_analytics.c.metric_brightness,
                active_project_prediction_analytics.c.metric_contrast,
                active_project_prediction_analytics.c.metric_sharpness,
                active_project_prediction_analytics.c.metric_red,
                active_project_prediction_analytics.c.metric_green,
                active_project_prediction_analytics.c.metric_blue,
                active_project_prediction_analytics.c.metric_label_border_closeness,
                active_project_prediction_analytics.c.metric_label_confidence,
                text(
                    """CASE
                        WHEN feature_hash == match_feature_hash THEN 1
                        ELSE 0
                    END AS true_positive
                """
                ),
            ]
        ).where(active_project_prediction_analytics.c.prediction_hash == prediction_hash)

        with self._engine.begin() as conn:
            df = pd.read_sql(stmt, conn)

        return df

    def get_images_metrics(self) -> pd.DataFrame:
        active_project_analytics_data = Table(
            "active_project_analytics_data", self._metadata, autoload_with=self._engine
        )
        stmt = select(active_project_analytics_data).where(
            active_project_analytics_data.c.project_hash == f"{self._project_hash}"
        )

        with self._engine.begin() as conn:
            df = pd.read_sql(stmt, conn)

        return df

    def get_or_add_tag(self, tag: str) -> UUID:
        with Session(self._engine) as sess:
            if tag in self._existing_tags:
                return self._existing_tags[tag]

            tag_hash = uuid4()
            new_tag = ProjectTag(
                tag_hash=tag_hash,
                name=tag,
                project_hash=self._project_hash,
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
                        project_hash=self._project_hash,
                        du_hash=du_item.du_hash,
                        frame=du_item.frame,
                        tag_hash=tag_hash,
                    )
                )
            sess.commit()
