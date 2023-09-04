import sys
import uuid
from copy import deepcopy
from typing import List

import orjson
import sqlalchemy
from fastapi import APIRouter, Body
from loguru import logger
from sqlalchemy.sql.expression import cast as sql_cast
from sqlmodel import Session, select
from tqdm import tqdm
from typing_extensions import Annotated

from encord_active.db.models import ProjectDataMetadata, ProjectDataUnitMetadata
from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.writer import StatisticsObserver
from encord_active.lib.db.merged_metrics import initialize_merged_metrics
from encord_active.lib.db.predictions import ClassificationPrediction
from encord_active.lib.metrics.metric import Metric, StatsMetadata
from encord_active.lib.metrics.types import (
    AnnotationType,
    DataType,
    EmbeddingType,
    MetricType,
)
from encord_active.lib.metrics.utils import load_available_metrics
from encord_active.lib.metrics.writer import CSVMetricWriter, CSVVideoAverageWrapper
from encord_active.lib.project.project_file_structure import ProjectFileStructure
from encord_active.server.dependencies import ProjectFileStructureDep, engine
from encord_active.server.routers.project2_engine import engine
from encord_active.server.utils import filtered_merged_metrics, load_project_metrics

logger.remove()
logger.add(
    sys.stderr,
    format="<e>{time}</e> | <m>{level}</m> | {message} | <e><d>{extra}</d></e>",
    colorize=True,
    level="DEBUG",
)


router = APIRouter(prefix="/{project}/predictions")


def find_unique_and_valid_attributes(
    project: ProjectFileStructure, predictions: list[ClassificationPrediction]
) -> set[tuple[str, str]]:
    classifications = orjson.loads(project.ontology.read_text()).get("classifications", [])
    radio_attributes = {
        (clf["featureNodeHash"], attr["featureNodeHash"])
        for clf in classifications
        for attr in clf.get("attributes", [])
        if attr["type"] == "radio"
    }
    res = set(
        (
            (p.classification.feature_hash, p.classification.attribute_hash)
            for p in predictions
            if (p.classification.feature_hash, p.classification.attribute_hash) in radio_attributes
        )
    )
    logger.debug("Found attributes via predictions", attributes=res)
    return res


class ClassificationModelConfidenceMetric(Metric):
    def __init__(
        self,
        classification_hash: str,
        attribute_hash: str,
        project: ProjectFileStructure,
        predictions: list[ClassificationPrediction],
        model_name: str,
    ):
        ontology = orjson.loads(project.ontology.read_text())

        self.attr = next(
            (
                attr
                for clf in ontology.get("classifications", [])
                for attr in clf.get("attributes", [])
                if attr["featureNodeHash"] == attribute_hash and clf["featureNodeHash"] == classification_hash
            ),
            None,
        )
        if self.attr is None:
            raise ValueError(f"Couldn't find attribute hash {attribute_hash} in the ontology")

        classification_name = self.attr.get("name")

        project_meta = project.load_project_meta()
        self.project_hash = uuid.UUID(project_meta["project_hash"])
        self.sampling_rate = project_meta.get("sampling_rate", 1) or 1

        self.predictions = [p for p in predictions if p.classification.attribute_hash == attribute_hash]

        short_description = long_description = f"Model confidence for classification question `{classification_name}`"
        super().__init__(
            title=f"Model Confidence - {model_name} - {classification_name}",
            short_description=short_description,
            long_description=long_description,
            metric_type=MetricType.SEMANTIC,
            data_type=DataType.IMAGE,
            annotation_type=AnnotationType.NONE,
            embedding_type=EmbeddingType.IMAGE,
            doc_url=None,
        )

    def execute(self, iterator: Iterator, writer: CSVVideoAverageWrapper):
        with Session(engine) as sess:
            valid_frames: set[tuple[uuid.UUID, int]] = set(
                sess.exec(
                    select(
                        ProjectDataUnitMetadata.du_hash,
                        ProjectDataUnitMetadata.frame,
                    ).where(
                        ProjectDataUnitMetadata.project_hash == self.project_hash,
                    )
                ).all()
            )
            du_to_lh = {
                k: v
                for k, v in sess.exec(
                    select(ProjectDataUnitMetadata.du_hash, ProjectDataMetadata.label_hash)
                    .join(
                        ProjectDataMetadata, onclause=ProjectDataMetadata.data_hash == ProjectDataUnitMetadata.data_hash
                    )
                    .where(
                        ProjectDataMetadata.project_hash == self.project_hash,
                        ProjectDataUnitMetadata.project_hash == self.project_hash,
                    )
                ).all()
            }

        logger.debug(f"Found {len(valid_frames)} frames and {len(du_to_lh)} label hashes")

        cnt = 0
        used_label_hashes = set()
        for pred in tqdm(self.predictions, desc="Reading predictions"):
            pred_du_hash = uuid.UUID(pred.data_hash)
            if (pred_du_hash, pred.frame) not in valid_frames:
                continue
            lh = du_to_lh.get(uuid.UUID(pred.data_hash))
            if lh is None:
                continue
            used_label_hashes.add(lh)
            key = f"{lh}_{pred_du_hash}_{pred.frame:05d}"
            writer.write(pred.confidence, key=key)
            cnt += 1
        logger.debug(f"Assigned {cnt} scores to {len(used_label_hashes)} label hashes")


@router.post("/classification")
def add_predictions(
    project: ProjectFileStructureDep,
    model_name: Annotated[str, Body()],
    predictions: Annotated[List[ClassificationPrediction], Body()],
):
    with logger.contextualize(project=project.project_dir, model_name=model_name):
        logger.debug("starting classification import")
        for clf, attr in find_unique_and_valid_attributes(project, predictions):
            metric = ClassificationModelConfidenceMetric(clf, attr, project, predictions, model_name=model_name)
            logger.debug(f"Running metric for {metric.metadata.title}")
            metric_name = metric.metadata.get_unique_name()
            with CSVMetricWriter(project.project_dir, None, prefix=metric_name) as wrapped:  # type: ignore
                wrapped_observer = StatisticsObserver()
                wrapped.attach(wrapped_observer)

                with CSVVideoAverageWrapper(wrapped) as wrapper:
                    metric.execute(None, wrapper)  # type: ignore

                    # wrapped metadata
                    wrapped_meta_file = (project.metrics / f"{wrapped.filename.stem}.meta.json").expanduser()
                    metric.metadata.stats = StatsMetadata.from_stats_observer(wrapped_observer)
                    wrapped_meta_file.write_text(metric.metadata.json())

                    # wrapper metadata
                    wrapper_meta_file = (project.metrics / f"{wrapper.filename.stem}.meta.json").expanduser()
                    metadata = deepcopy(metric.metadata)
                    metadata.title = f"{metric.metadata.title} Average"
                    metadata.embedding_type = EmbeddingType.VIDEO
                    wrapper_meta_file.write_text(metadata.json())

        # app_import_predictions(_project, predictions, run_metrics=False)
        logger.debug("Initializing merged metrics")
        initialize_merged_metrics(
            project
        )  # This might work but not sure if it's enough might need to use `ensure_safe_project`?
        # migrate_disk_to_db(project, delete_existing_project=True)
        logger.debug("Clearing cache")
        load_project_metrics.cache_clear()
        filtered_merged_metrics.cache_clear()
