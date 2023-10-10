import random
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from encord_active.analysis.config import create_analysis, default_torch_device
from encord_active.analysis.executor import SimpleExecutor
from encord_active.analysis.reductions.op import (
    apply_embedding_reduction,
    create_reduction,
    serialize_reduction,
)
from encord_active.db.models import (
    Project,
    ProjectCollaborator,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectPrediction,
    ProjectPredictionDataMetadata,
    ProjectPredictionDataUnitMetadata,
)
from encord_active.db.models.project_embedding_reduction import (
    EmbeddingReductionType,
    ProjectEmbeddingReduction,
)
from encord_active.db.models.project_prediction_analytics_extra import (
    ProjectPredictionAnalyticsExtra,
)
from encord_active.db.models.project_prediction_analytics_reduced import (
    ProjectPredictionAnalyticsReduced,
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


def import_prediction(engine: Engine, database_dir: Path, ssh_key: str, prediction: PredictionImportSpec) -> None:
    prediction_hash = prediction.prediction.prediction_hash
    coco_timestamp = datetime.now()

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
        project_collaborator_list = sess.exec(
            select(ProjectCollaborator).where(ProjectCollaborator.project_hash == project_hash)
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
            created_at=coco_timestamp,
            last_edited_at=coco_timestamp,
            object_answers={},
            classification_answers={},
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
            data_title=du_meta.data_title,
            data_type=du_meta.data_type,
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
        ground_truth_data_meta=project_data_meta_list,
        ground_truth_annotation_meta=project_du_meta_list,
        predicted_data_meta=prediction_data_metadata,
        predicted_annotation_meta=prediction_data_unit_metadata,
        collaborators=project_collaborator_list,
        database_dir=database_dir,
        project_hash=project_hash,
        prediction_hash=prediction_hash,
        project_ssh_key=ssh_key if project.remote else None,
        project_ontology=project.ontology,
    )
    (
        prediction_analytics,
        prediction_analytics_extra,
        prediction_analytics_derived,
        prediction_analytics_false_negatives,
        new_collaborators,
    ) = res

    prediction_db_data_meta = [
        ProjectPredictionDataMetadata(
            prediction_hash=prediction_hash,
            data_hash=d.data_hash,
            project_hash=project_hash,
            label_hash=d.label_hash,
            created_at=d.created_at,
            last_edited_at=d.last_edited_at,
            object_answers=d.object_answers,
            classification_answers=d.classification_answers,
        )
        for d in prediction_data_metadata
    ]
    prediction_db_data_unit_meta = [
        ProjectPredictionDataUnitMetadata(
            prediction_hash=prediction_hash,
            du_hash=du.du_hash,
            frame=du.frame,
            project_hash=project_hash,
            data_hash=du.data_hash,
            objects=du.objects,
            classifications=du.classifications,
        )
        for du in prediction_data_unit_metadata
    ]

    reduction_train_samples = random.choices(prediction_analytics_extra, k=min(len(prediction_analytics_extra), 50_000))
    reduction_train_samples.sort(key=lambda x: (x.du_hash, x.frame, x.annotation_hash))
    reduction = create_reduction(
        EmbeddingReductionType.UMAP,
        train_samples=[
            np.frombuffer(sample.embedding_clip or b"", dtype=np.float64) for sample in reduction_train_samples
        ],
    )
    reduction_hash = sess.exec(
        select(ProjectEmbeddingReduction.reduction_hash).where(ProjectEmbeddingReduction.project_hash == project_hash)
    ).fetchall()[0]

    reduced_annotation_clip_raw = apply_embedding_reduction(
        [np.frombuffer(sample.embedding_clip or b"", dtype=np.float64) for sample in prediction_analytics_extra],
        reduction,
    )
    prediction_analytics_reduced = [
        ProjectPredictionAnalyticsReduced(
            reduction_hash=reduction_hash,
            prediction_hash=prediction_hash,
            project_hash=project_hash,
            du_hash=extra.du_hash,
            frame=extra.frame,
            annotation_hash=extra.annotation_hash,
            x=x,
            y=y,
        )
        for (extra, (x, y)) in zip(prediction_analytics_extra, reduced_annotation_clip_raw)
    ]

    # Store to the database
    with Session(engine) as sess:
        sess.add(prediction.prediction)
        sess.add_all(new_collaborators)
        sess.add_all(prediction_db_data_meta)
        sess.add_all(prediction_db_data_unit_meta)
        sess.add_all(prediction_analytics)
        sess.add_all(prediction_analytics_extra)
        sess.add_all(prediction_analytics_reduced)
        sess.add_all(prediction_analytics_derived)
        sess.add_all(prediction_analytics_false_negatives)
        sess.commit()
