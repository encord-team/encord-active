import base64
import enum
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional, Type, TypeVar

from pydantic import BaseModel, Field
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from encord_active.db.local_data import db_uri_to_local_file_path
from encord_active.db.models import (
    Project,
    ProjectAnnotationAnalytics,
    ProjectAnnotationAnalyticsDerived,
    ProjectAnnotationAnalyticsExtra,
    ProjectAnnotationAnalyticsReduced,
    ProjectCollaborator,
    ProjectDataAnalytics,
    ProjectDataAnalyticsDerived,
    ProjectDataAnalyticsExtra,
    ProjectDataAnalyticsReduced,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectEmbeddingIndex,
    ProjectEmbeddingReduction,
    ProjectImportMetadata,
    ProjectPrediction,
    ProjectPredictionAnalytics,
    ProjectPredictionAnalyticsDerived,
    ProjectPredictionAnalyticsExtra,
    ProjectPredictionAnalyticsFalseNegatives,
    ProjectPredictionAnalyticsReduced,
    ProjectPredictionDataMetadata,
    ProjectPredictionDataUnitMetadata,
    ProjectTag,
    ProjectTaggedAnnotation,
    ProjectTaggedDataUnit,
    ProjectTaggedPrediction,
)


class WholeProjectModelV1(BaseModel):
    version: Literal["v1"]
    prediction: List[ProjectPrediction]
    prediction_analytics: List[ProjectPredictionAnalytics]
    prediction_analytics_derived: List[ProjectPredictionAnalyticsDerived]
    prediction_analytics_extra: List[ProjectPredictionAnalyticsExtra]
    prediction_analytics_fn: List[ProjectPredictionAnalyticsFalseNegatives]
    prediction_analytics_reduced: List[ProjectPredictionAnalyticsReduced]
    prediction_data: List[ProjectPredictionDataMetadata]
    prediction_data_units: List[ProjectPredictionDataUnitMetadata]
    project: List[Project]
    project_analytics_annotation: List[ProjectAnnotationAnalytics]
    project_analytics_annotation_derived: List[ProjectAnnotationAnalyticsDerived]
    project_analytics_annotation_extra: List[ProjectAnnotationAnalyticsExtra]
    project_analytics_annotation_reduced: List[ProjectAnnotationAnalyticsReduced]
    project_analytics_data: List[ProjectDataAnalytics]
    project_analytics_data_derived: List[ProjectDataAnalyticsDerived]
    project_analytics_data_extra: List[ProjectDataAnalyticsExtra]
    project_analytics_data_reduced: List[ProjectDataAnalyticsReduced]
    project_collaborator: List[ProjectCollaborator]
    project_data: List[ProjectDataMetadata]
    project_data_units: List[ProjectDataUnitMetadata]
    project_embedding_index: List[ProjectEmbeddingIndex]
    project_embedding_reduction: List[ProjectEmbeddingReduction]
    project_import: List[ProjectImportMetadata] = Field(default=[])
    project_tagged_annotation: List[ProjectTaggedAnnotation] = Field(default=[])
    project_tagged_data: List[ProjectTaggedDataUnit] = Field(default=[])
    project_tagged_prediction: List[ProjectTaggedPrediction] = Field(default=[])
    project_tags: List[ProjectTag] = Field(default=[])


T = TypeVar(
    "T",
    ProjectPrediction,
    ProjectPredictionAnalytics,
    ProjectPredictionAnalyticsDerived,
    ProjectPredictionAnalyticsExtra,
    ProjectPredictionAnalyticsFalseNegatives,
    ProjectPredictionAnalyticsReduced,
    ProjectPredictionDataMetadata,
    ProjectPredictionDataUnitMetadata,
    Project,
    ProjectAnnotationAnalyticsExtra,
    ProjectAnnotationAnalyticsDerived,
    ProjectAnnotationAnalytics,
    ProjectDataAnalytics,
    ProjectDataAnalyticsDerived,
    ProjectDataAnalyticsExtra,
    ProjectAnnotationAnalyticsReduced,
    ProjectDataAnalyticsReduced,
    ProjectDataUnitMetadata,
    ProjectDataMetadata,
    ProjectCollaborator,
    ProjectEmbeddingIndex,
    ProjectEmbeddingReduction,
    ProjectImportMetadata,
    ProjectTaggedAnnotation,
    ProjectTaggedDataUnit,
    ProjectTaggedPrediction,
    ProjectTag,
)


def serialize_encoder(value):
    if value is None:
        return None
    if isinstance(value, (str, int, float)):
        return value
    if isinstance(value, (uuid.UUID, datetime)):
        return str(value)
    if isinstance(value, bytes):
        return base64.b85encode(value).decode("utf-8")
    if isinstance(value, enum.Enum):
        return value.value
    raise ValueError(f"{type(value)}")


def serialize_whole_project_state(engine: Engine, database_dir: Path, project_hash: uuid.UUID, folder: Path) -> None:
    with Session(engine) as sess:

        def fetch_all(table: Type[T]) -> List[T]:
            return sess.exec(select(table).where(table.project_hash == project_hash)).fetchall()

        whole_project_table = WholeProjectModelV1(
            version="v1",
            prediction=fetch_all(ProjectPrediction),
            prediction_analytics=fetch_all(ProjectPredictionAnalytics),
            prediction_analytics_derived=fetch_all(ProjectPredictionAnalyticsDerived),
            prediction_analytics_extra=fetch_all(ProjectPredictionAnalyticsExtra),
            prediction_analytics_fn=fetch_all(ProjectPredictionAnalyticsFalseNegatives),
            prediction_analytics_reduced=fetch_all(ProjectPredictionAnalyticsReduced),
            prediction_data=fetch_all(ProjectPredictionDataMetadata),
            prediction_data_units=fetch_all(ProjectPredictionDataUnitMetadata),
            project=fetch_all(Project),
            project_analytics_annotation=fetch_all(ProjectAnnotationAnalytics),
            project_analytics_annotation_derived=fetch_all(ProjectAnnotationAnalyticsDerived),
            project_analytics_annotation_extra=fetch_all(ProjectAnnotationAnalyticsExtra),
            project_analytics_annotation_reduced=fetch_all(ProjectAnnotationAnalyticsReduced),
            project_analytics_data=fetch_all(ProjectDataAnalytics),
            project_analytics_data_derived=fetch_all(ProjectDataAnalyticsDerived),
            project_analytics_data_extra=fetch_all(ProjectDataAnalyticsExtra),
            project_analytics_data_reduced=fetch_all(ProjectDataAnalyticsReduced),
            project_collaborator=fetch_all(ProjectCollaborator),
            project_data=fetch_all(ProjectDataMetadata),
            project_data_units=fetch_all(ProjectDataUnitMetadata),
            project_embedding_index=fetch_all(ProjectEmbeddingIndex),
            project_embedding_reduction=fetch_all(ProjectEmbeddingReduction),
            project_import=fetch_all(ProjectImportMetadata),
            project_tagged_annotation=fetch_all(ProjectTaggedAnnotation),
            project_tagged_data=fetch_all(ProjectTaggedDataUnit),
            project_tagged_prediction=fetch_all(ProjectTaggedPrediction),
            project_tags=fetch_all(ProjectTag),
        )

    # Save (& change db schema) all local files
    all_local_files = {
        du.data_uri
        for du in whole_project_table.project_data_units
        if du.data_uri is not None and not du.data_uri.startswith("http://") and not du.data_uri.startswith("https://")
    }
    local_file_map = {uri: uuid.uuid4() for uri in all_local_files}

    # Remap
    def _transform(du: ProjectDataUnitMetadata) -> Optional[str]:
        if du.data_uri is not None:
            new_uuid_val = local_file_map.get(du.data_uri, None)
            if new_uuid_val is not None:
                return f"relative://local-data/{folder.name}/{new_uuid_val}"
        return du.data_uri

    whole_project_table.project_data_units = [
        ProjectDataUnitMetadata(**{k: v for k, v in dict(du).items() if k != "data_uri"}, data_uri=_transform(du))
        for du in whole_project_table.project_data_units
    ]

    # Actually save & generate the serialized state
    folder.mkdir(parents=True, exist_ok=False)
    for old_uri, new_uuid in local_file_map.items():
        old_uri_path = db_uri_to_local_file_path(old_uri, database_dir)
        if old_uri_path is None:
            raise ValueError(f"Corrupt uri path: {old_uri_path}")
        shutil.copyfile(old_uri_path, folder / str(new_uuid))

    # Serialize whole state
    (folder / "db.json").write_text(whole_project_table.json(encoder=serialize_encoder, indent=1), "utf-8")
