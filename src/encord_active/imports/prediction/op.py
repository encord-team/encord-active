import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import List

from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from encord_active.analysis.config import create_analysis, default_torch_device
from encord_active.analysis.executor import SimpleExecutor
from encord_active.db.models import (
    Project,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectPrediction,
)


class ProjectPredictedDataUnit(BaseModel):
    du_hash: uuid.UUID
    frame: int

    # Same format as encord-data-unit metadata
    objects: list
    classifications: list


@dataclass
class PredictionImportSpec:
    prediction: ProjectPrediction
    du_prediction_list: List[ProjectPredictedDataUnit]


def import_prediction(engine: Engine, database_dir: Path, prediction: PredictionImportSpec) -> None:
    prediction_hash = prediction.prediction.prediction_hash

    # Group by data unit/frame
    du_prediction_lookup = {}
    for du_prediction in prediction.du_prediction_list:
        if (du_prediction.du_hash, du_prediction.frame) in du_prediction_lookup:
            raise ValueError(f"Duplicate prediction grouping: {du_prediction.du_hash}, {du_prediction.frame}")
        du_prediction_lookup[(du_prediction.du_hash, du_prediction.frame)] = du_prediction

    # Convert to format for execution in the metric engine.
    project_hash = prediction.prediction.project_hash
    with Session(engine) as sess:
        project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if project is None:
            raise ValueError(f"Project for prediction import does not exist ({project_hash})!!")
        project_data_meta_list = sess.exec(
            select(  # type: ignore
                ProjectDataMetadata.data_hash,
                ProjectDataMetadata.dataset_hash,
                ProjectDataMetadata.num_frames,
                ProjectDataMetadata.frames_per_second,
                ProjectDataMetadata.dataset_title,
                ProjectDataMetadata.data_title,
                ProjectDataMetadata.data_type,
            ).where(ProjectDataMetadata.project_hash == project_hash)
        ).fetchall()
        project_du_meta_list = sess.exec(
            select(ProjectDataUnitMetadata).where(ProjectDataUnitMetadata.project_hash == project_hash)
        ).fetchall()

    prediction_data_metadata = [
        ProjectDataMetadata(
            project_hash=project_hash,
            data_hash=data_hash,
            label_hash=uuid.uuid4(),
            dataset_hash=dataset_hash,
            num_frames=num_frames,
            frames_per_second=frames_per_second,
            dataset_title=dataset_title,
            data_title=data_title,
            data_type=data_type,
            label_row_json={},
        )
        for data_hash, dataset_hash, num_frames, frames_per_second, dataset_title, data_title, data_type in project_data_meta_list
    ]

    prediction_data_unit_metadata = [
        ProjectDataUnitMetadata(
            project_hash=project_hash,
            du_hash=du_meta.du_hash,
            frame=du_meta.frame,
            data_hash=du_meta.data_hash,
            width=du_meta.width,
            height=du_meta.height,
            data_uri=du_meta.data_uri,
            data_uri_is_video=du_meta.data_uri_is_video,
            objects=du_prediction_lookup[(du_meta.du_hash, du_meta.frame)].objects
            if (du_meta.du_hash, du_meta.frame) in du_prediction_lookup
            else [],
            classifications=du_prediction_lookup.pop((du_meta.du_hash, du_meta.frame)).classifications
            if (du_meta.du_hash, du_meta.frame) in du_prediction_lookup
            else [],
        )
        for du_meta in project_du_meta_list
    ]
    if len(du_prediction_lookup) != 0:
        raise ValueError(
            f"Prediction found referencing invalid du_hash, frame pair(s): {list(du_prediction_lookup.keys())}"
        )

    # Execute metric engine.
    metric_engine = SimpleExecutor(create_analysis(default_torch_device()))
    res = metric_engine.execute_prediction_from_db(
        data_meta=prediction_data_metadata,
        ground_truth_annotation_meta=project_du_meta_list,
        predicted_annotation_meta=prediction_data_unit_metadata,
        database_dir=database_dir,
        project_hash=project_hash,
        prediction_hash=prediction_hash,
        project_ssh_path=project.project_remote_ssh_key_path,
    )
    prediction_analytics, prediction_analytics_extra, prediction_analytics_false_negatives = res

    # Store to the database
    with Session(engine) as sess:
        sess.add(prediction.prediction)
        sess.add_all(prediction_analytics)
        sess.add_all(prediction_analytics_extra)
        sess.add_all(prediction_analytics_false_negatives)
        sess.commit()
