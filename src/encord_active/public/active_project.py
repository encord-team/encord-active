from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Optional, Type, Union
from uuid import UUID, uuid4

import pandas as pd
from encord.objects import OntologyStructure
from sqlalchemy import Integer, func
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from encord_active.db.models import (
    Project,
    ProjectAnnotationAnalytics,
    ProjectDataAnalytics,
    ProjectDataUnitMetadata,
    ProjectPrediction,
    ProjectPredictionAnalytics,
    ProjectTag,
    ProjectTaggedDataUnit,
    get_engine,
)
from encord_active.lib.common.data_utils import url_to_file_path

_P = Project
_T = ProjectTag


@dataclass
class DataUnitItem:
    du_hash: UUID
    frame: int


AnalyticsModel = Union[Type[ProjectDataAnalytics], Type[ProjectPredictionAnalytics], Type[ProjectAnnotationAnalytics]]


def get_active_engine(path_to_db: Union[str, Path]) -> Engine:
    path = Path(path_to_db) if isinstance(path_to_db, str) else path_to_db
    return get_engine(path, use_alembic=False)


class ActiveProject:
    @classmethod
    def from_db_file(cls, db_file_path: Union[str, Path], project_name: str):
        path = Path(db_file_path) if isinstance(db_file_path, str) else db_file_path
        engine = get_active_engine(db_file_path)
        return ActiveProject(engine, project_name, root_path=path.parent)

    def __init__(self, engine: Optional[Engine], project_name: str, root_path: Optional[Path] = None):
        self._engine = engine
        self._project_name = project_name
        self._root_path = root_path

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

    def _join_path_statement(self, stmt, base_model: AnalyticsModel):
        stmt = stmt.add_columns(ProjectDataUnitMetadata.data_uri, ProjectDataUnitMetadata.data_uri_is_video)
        stmt = stmt.join(
            ProjectDataUnitMetadata,
            onclause=(base_model.du_hash == ProjectDataUnitMetadata.du_hash)
            & (base_model.frame == ProjectDataUnitMetadata.frame),
        ).where(ProjectDataUnitMetadata.project_hash == self.project_hash)

        def transform(df):
            if self._root_path is None:
                raise ValueError(f"Root path is not set. Provide it in the constructor or use `from_db_file`")
            df["data_uri"] = df.data_uri.map(partial(url_to_file_path, project_dir=self._root_path))
            return df

        return stmt, transform

    def _join_data_tags_statement(self, stmt, base_model: AnalyticsModel, group_by_cols=None):
        stmt = stmt.add_columns(("[" + func.group_concat('"' + ProjectTag.name + '"', ", ") + "]").label("data_tags"))
        stmt = (
            stmt.join(
                ProjectTaggedDataUnit,
                onclause=(base_model.du_hash == ProjectTaggedDataUnit.du_hash)
                & (base_model.frame == ProjectTaggedDataUnit.frame),
            )
            .join(ProjectTag, onclause=ProjectTag.tag_hash == ProjectTaggedDataUnit.tag_hash)
            .where(
                ProjectTag.project_hash == self.project_hash, ProjectTaggedDataUnit.project_hash == self.project_hash
            )
            .group_by(base_model.du_hash, base_model.frame, *(group_by_cols or []))
        )

        def transform(df):
            df["data_tags"] = df.data_tags.map(eval)
            return df

        return stmt, transform

    def get_prediction_metrics(self, include_data_uris: bool = False, include_data_tags: bool = False) -> pd.DataFrame:
        """
        Returns a pandas data frame with all the prediction metrics.

        Args:
            include_data_uris: If set to true, the data frame will contain a data_uri column containing the path to the image file.
            include_data_tags: If set to true, the data frame will contain a data_tags column containing a list of tags for the underlying image.
                Disclaimer. We take no measures here to counteract SQL injections so avoid using "funky" characters like '"\\/` in your tag names.

        """
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

            transforms = []
            if include_data_uris:
                stmt, transform = self._join_path_statement(stmt, P)
                transforms.append(transform)
            if include_data_tags:
                stmt, transform = self._join_data_tags_statement(stmt, P, group_by_cols=[P.object_hash])
                transforms.append(transform)

            df = pd.DataFrame(sess.exec(stmt).all())
            df.columns = list(sess.exec(stmt).keys())

            for transform in transforms:
                df = transform(df)

        return df

    def get_images_metrics(self, *, include_data_uris: bool = False, include_data_tags: bool = True) -> pd.DataFrame:
        """
        Returns a pandas data frame with all the prediction metrics.

        Args:
            include_data_uris: If set to true, the data frame will contain a data_uri column containing the path to the image file.
            include_data_tags: If set to true, the data frame will contain a data_tags column containing a list of tags for the underlying image.
                Disclaimer. We take no measures here to counteract SQL injections so avoid using "funky" characters like '"\\/` in your tag names.
        """
        with Session(self._engine) as sess:
            stmt = select(*[c for c in ProjectDataAnalytics.__table__.c]).where(  # type: ignore -- hack to get all columns without the pydantic model
                ProjectDataAnalytics.project_hash == self.project_hash
            )
            transforms = []
            if include_data_uris:
                stmt, transform = self._join_path_statement(stmt, ProjectDataAnalytics)
                transforms.append(transform)
            if include_data_tags:
                stmt, transform = self._join_data_tags_statement(stmt, ProjectDataAnalytics)
                transforms.append(transform)
            image_metrics = sess.exec(stmt).all()
            df = pd.DataFrame(image_metrics)
            df.columns = list(sess.execute(stmt).keys())

            for transform in transforms:
                df = transform(df)

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
