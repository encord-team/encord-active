import json
import shutil
import uuid
from enum import Enum
from functools import partial
from typing import Annotated, Dict, List, Literal, Optional, Union, cast, overload
from uuid import UUID

import pandas as pd
from cachetools import LRUCache, cached
from encord.orm.project import (
    CopyDatasetAction,
    CopyDatasetOptions,
    CopyLabelsOptions,
    ReviewApprovalState,
)
from fastapi import (
    APIRouter,
    BackgroundTasks,
    Body,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
)
from fastapi.responses import ORJSONResponse
from natsort import natsorted
from pandera.typing import DataFrame
from pydantic import BaseModel
from sqlalchemy import delete, tuple_
from sqlalchemy.sql.operators import in_op
from sqlmodel import Session, select
from starlette.responses import FileResponse
from starlette.status import HTTP_403_FORBIDDEN

from encord_active.cli.app_config import app_config
from encord_active.cli.utils.server import ensure_safe_project
from encord_active.db.models import (
    Project,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
    ProjectTag,
    ProjectTaggedAnnotation,
    ProjectTaggedDataUnit,
)
from encord_active.db.scripts.delete_project import delete_project_from_db
from encord_active.db.scripts.migrate_disk_to_db import migrate_disk_to_db
from encord_active.lib.common.data_utils import url_to_file_path
from encord_active.lib.common.filtering import Filters, Range, apply_filters
from encord_active.lib.common.utils import (
    DataHashMapping,
    IndexOrSeries,
    partial_column,
)
from encord_active.lib.db.connection import DBConnection, PrismaConnection
from encord_active.lib.db.helpers.tags import GroupedTags
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.embeddings.dimensionality_reduction import get_2d_embedding_data
from encord_active.lib.embeddings.types import Embedding2DSchema, Embedding2DScoreSchema
from encord_active.lib.encord.actions import (
    DatasetUniquenessError,
    EncordActions,
    replace_db_uids,
)
from encord_active.lib.encord.project_sync import (
    LabelRowDataUnit,
    copy_filtered_data,
    copy_image_data_unit_json,
    copy_label_row_meta_json,
    copy_project_meta,
    create_filtered_db,
    create_filtered_embeddings,
    create_filtered_metrics,
    replace_uids,
)
from encord_active.lib.encord.utils import get_encord_project
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.metrics.utils import (
    MetricScope,
    filter_none_empty_metrics,
    get_embedding_type,
)
from encord_active.lib.model_predictions.reader import (
    check_model_prediction_availability,
    get_model_prediction_by_id,
    get_model_predictions,
    read_prediction_files,
)
from encord_active.lib.model_predictions.types import (
    ClassificationOutcomeType,
    ClassificationPredictionMatchSchema,
    LabelMatchSchema,
    ObjectDetectionOutcomeType,
    PredictionMatchSchema,
    PredictionsFilters,
)
from encord_active.lib.model_predictions.writer import MainPredictionType
from encord_active.lib.premium.model import CLIPQuery, TextQuery
from encord_active.lib.premium.querier import Querier
from encord_active.lib.project.metadata import fetch_project_meta, update_project_meta
from encord_active.lib.project.project_file_structure import ProjectFileStructure
from encord_active.lib.project.sandbox_projects.sandbox_projects import (
    available_prebuilt_projects,
    fetch_prebuilt_project,
)
from encord_active.server.dependencies import (
    ProjectFileStructureDep,
    engine,
    verify_premium,
)
from encord_active.server.routers.project2 import get_all_projects
from encord_active.server.settings import get_settings
from encord_active.server.utils import (
    filtered_merged_metrics,
    get_similarity_finder,
    load_project_metrics,
    to_item,
)

router = APIRouter(
    prefix="/projects",
    tags=["projects"],
)


@router.post("/{project}/item_ids_by_metric", response_class=ORJSONResponse)
def read_item_ids(
    project: ProjectFileStructureDep,
    scope: Annotated[MetricScope, Body()],
    sort_by_metric: Annotated[str, Body()],
    filters: Filters = Filters(),
    ascending: Annotated[bool, Body()] = True,
    ids: Annotated[Optional[list[str]], Body()] = None,
):
    merged_metrics = filtered_merged_metrics(project, filters, scope)

    if scope == MetricScope.PREDICTION:
        if filters.prediction_filters is None:
            raise HTTPException(
                status_code=422, detail='Filters must contain "prediction_filters" when scope is "prediction"'
            )
        df, _ = get_model_predictions(project, filters.prediction_filters)
        df = apply_filters(df, filters, project, scope)

        if filters.prediction_filters.outcome == ObjectDetectionOutcomeType.FALSE_NEGATIVES:
            sort_by_metric = sort_by_metric.replace("(P)", "(O)")

        column = [col for col in df.columns if col.lower() == sort_by_metric.lower()][0]
        df = df[partial_column(df.index, 3).isin(partial_column(merged_metrics.index, 3).unique())]
        if filters.prediction_filters.outcome == ObjectDetectionOutcomeType.FALSE_NEGATIVES:
            df = df[df.index.isin(merged_metrics.index)]
    else:
        column = [col for col in merged_metrics.columns if col.lower() == sort_by_metric.lower()][0]
        df = merged_metrics

    res: pd.DataFrame = df[[column]].dropna().sort_values(by=[column], ascending=ascending)
    if ids:
        if filters.prediction_filters and filters.prediction_filters.type == MainPredictionType.OBJECT:
            res = res[partial_column(res.index, 3).isin(ids)]
        else:
            res = res[res.index.isin(ids)]
    res = res.reset_index().rename({"identifier": "id", column: "value"}, axis=1)

    return ORJSONResponse(res[["id", "value"]].to_dict("records"))


@router.get("/{project}/tagged_items")
def tagged_items(project: ProjectFileStructureDep):
    project_hash = uuid.UUID(project.load_project_meta()["project_hash"])
    identifier_tags: dict[str, GroupedTags] = {}
    with Session(engine) as sess:
        data_tags = sess.exec(
            select(
                ProjectDataMetadata.label_hash,
                ProjectTaggedDataUnit.du_hash,
                ProjectTaggedDataUnit.frame,
                ProjectTag.name,
            ).where(
                ProjectTaggedDataUnit.tag_hash == ProjectTag.tag_hash,
                ProjectTaggedDataUnit.project_hash == project_hash,
                ProjectTag.project_hash == project_hash,
                ProjectDataMetadata.project_hash == project_hash,
                ProjectDataUnitMetadata.project_hash == project_hash,
                ProjectDataUnitMetadata.data_hash == ProjectDataMetadata.data_hash,
                ProjectDataUnitMetadata.du_hash == ProjectTaggedDataUnit.du_hash,
                ProjectDataUnitMetadata.frame == ProjectTaggedDataUnit.frame,
            )
        ).all()
        for label_hash, du_hash, frame, tag in data_tags:
            key = f"{label_hash}_{du_hash}_{frame:05d}"
            identifier_tags.setdefault(key, GroupedTags(data=[], label=[]))["data"].append(tag)
        label_tags = sess.exec(
            select(  # type: ignore
                ProjectDataMetadata.label_hash,
                ProjectTaggedAnnotation.du_hash,
                ProjectTaggedAnnotation.frame,
                ProjectTaggedAnnotation.object_hash,
                ProjectTag.name,
            ).where(
                ProjectTaggedAnnotation.tag_hash == ProjectTag.tag_hash,
                ProjectTaggedAnnotation.project_hash == project_hash,
                ProjectTag.project_hash == project_hash,
                ProjectDataUnitMetadata.project_hash == project_hash,
                ProjectDataUnitMetadata.data_hash == ProjectDataMetadata.data_hash,
                ProjectDataUnitMetadata.du_hash == ProjectTaggedAnnotation.du_hash,
                ProjectDataUnitMetadata.frame == ProjectTaggedAnnotation.frame,
            )
        ).all()
        for label_hash, du_hash, frame, annotation_hash, tag in label_tags:
            data_key = f"{label_hash}_{du_hash}_{frame:05d}"
            label_key = f"{data_key}_{annotation_hash}"
            du_tags = identifier_tags.setdefault(data_key, GroupedTags(data=[], label=[]))
            if tag not in du_tags["label"]:
                du_tags["label"].append(tag)
            identifier_tags.setdefault(label_key, GroupedTags(data=du_tags["data"], label=[]))["label"].append(tag)
    return identifier_tags


@router.get("/{project}/local-fs/{lr_hash}/{du_hash}/{frame}")
def server_local_fs_file(project: ProjectFileStructureDep, lr_hash: str, du_hash: str, frame: int):
    label_row_structure = project.label_row_structure(lr_hash)
    data_opt = next(label_row_structure.iter_data_unit(du_hash, int(frame)), None) or next(
        label_row_structure.iter_data_unit(du_hash, None), None
    )
    if data_opt is not None:
        signed_url = data_opt.signed_url
        file_path = url_to_file_path(signed_url, label_row_structure.project.project_dir)
        if file_path is not None:
            return FileResponse(file_path)

    debug_id = f"{lr_hash}_{du_hash}_{frame}"
    raise HTTPException(
        status_code=404, detail=f'Local resource with id "{debug_id}" was not found for project "{project}"'
    )


def append_tags_to_row(project: ProjectFileStructureDep, row: dict):
    project_hash = uuid.UUID(project.load_project_meta()["project_hash"])
    _, du_hash_str, frame_str, *annotation_hashes = row["identifier"].split("_")
    du_hash = uuid.UUID(du_hash_str)
    frame = int(frame_str)

    with Session(engine) as sess:
        data_tags = sess.exec(
            select(ProjectTag.name)
            .join(ProjectTaggedDataUnit)
            .where(
                ProjectTaggedDataUnit.project_hash == project_hash,
                ProjectTaggedDataUnit.du_hash == du_hash,
                ProjectTaggedDataUnit.frame == frame,
                ProjectTaggedDataUnit.tag_hash == ProjectTag.tag_hash,
            )
        ).all()

    label_tags = []
    with Session(engine) as sess:
        if annotation_hashes:
            for annotation_hash in annotation_hashes:
                label_tags += sess.exec(
                    select(ProjectTag.name)
                    .join(ProjectTaggedAnnotation)
                    .where(
                        ProjectTaggedAnnotation.project_hash == project_hash,
                        ProjectTaggedAnnotation.du_hash == du_hash,
                        ProjectTaggedAnnotation.frame == frame,
                        ProjectTaggedAnnotation.object_hash == annotation_hash,
                        ProjectTaggedAnnotation.tag_hash == ProjectTag.tag_hash,
                    )
                ).all()
        else:
            label_tags += sess.exec(
                select(ProjectTag.name)
                .join(ProjectTaggedAnnotation)
                .where(
                    ProjectTaggedAnnotation.project_hash == project_hash,
                    ProjectTaggedAnnotation.du_hash == du_hash,
                    ProjectTaggedAnnotation.frame == frame,
                    ProjectTaggedAnnotation.tag_hash == ProjectTag.tag_hash,
                )
            ).unique()

    row["tags"] = {"data": data_tags, "label": label_tags}


@router.get("/{project}/items/{id:path}")
def read_item(project: ProjectFileStructureDep, id: str, iou: Optional[float] = None):
    lr_hash, du_hash, frame, *object_hash = id.split("_")

    row = get_model_prediction_by_id(project, id, iou)

    if row:
        return to_item(row, project, lr_hash, du_hash, frame, object_hash[0] if len(object_hash) else None)

    with DBConnection(project) as conn:
        rows = MergedMetrics(conn).get_row(id).dropna(axis=1).to_dict("records")

        if not rows:
            raise HTTPException(status_code=404, detail=f'Item with id "{id}" was not found for project "{project}"')

        row = rows[0]

    append_tags_to_row(project, row)
    return to_item(row, project, lr_hash, du_hash, frame)


class ItemTags(BaseModel):
    id: str
    grouped_tags: GroupedTags


@overload
def _get_selector(item: ItemTags, annotation_hash: Literal[False]) -> Optional[tuple[uuid.UUID, int]]:
    ...


@overload
def _get_selector(item: ItemTags, annotation_hash: Literal[True]) -> Optional[tuple[uuid.UUID, int, str]]:
    ...


def _get_selector(
    item: ItemTags, annotation_hash: bool = False
) -> Optional[Union[tuple[uuid.UUID, int], tuple[uuid.UUID, int, str]]]:
    _, du_hash_str, frame_str, *annotation_hashes = item.id.split("_")
    base = (uuid.UUID(du_hash_str), int(frame_str))
    if annotation_hash:
        if not annotation_hashes:
            return None
        return (*base, annotation_hashes[0])
    return base


@router.put("/{project}/item_tags")
def tag_items(project: ProjectFileStructureDep, payload: List[ItemTags]):
    with Session(engine) as sess:
        project_hash = uuid.UUID(project.load_project_meta()["project_hash"])

        project_tags = {
            r[0]: r[1]
            for r in sess.exec(
                select(ProjectTag.name, ProjectTag.tag_hash).where(ProjectTag.project_hash == project_hash)
            ).all()
        }

        def _get_or_create_tag_hash(name: str) -> uuid.UUID:
            tag_hash = project_tags.get(name)
            if tag_hash is not None:
                return tag_hash

            new_tag_hash = uuid.uuid4()
            sess.add(
                ProjectTag(
                    tag_hash=new_tag_hash,
                    project_hash=project_hash,
                    name=name,
                    description="",
                )
            )
            project_tags[name] = new_tag_hash
            return new_tag_hash

        data_selectors = set(filter(None, map(partial(_get_selector, annotation_hash=False), payload)))
        existing_data_tag_tuples = sess.exec(
            select(ProjectTaggedDataUnit.du_hash, ProjectTaggedDataUnit.frame, ProjectTaggedDataUnit.tag_hash).where(
                ProjectTaggedDataUnit.project_hash == project_hash,
                in_op(tuple_(ProjectTaggedDataUnit.du_hash, ProjectTaggedDataUnit.frame), data_selectors),
            )
        ).all()
        existing_data_tags: dict[tuple[uuid.UUID, int], set[uuid.UUID]] = {}
        for du_hash, frame, tag_hash in existing_data_tag_tuples:
            existing_data_tags.setdefault((du_hash, frame), set()).add(tag_hash)

        label_selectors = set(filter(None, map(partial(_get_selector, annotation_hash=True), payload)))
        existing_label_tag_tuples = sess.exec(
            select(
                ProjectTaggedAnnotation.du_hash,
                ProjectTaggedAnnotation.frame,
                ProjectTaggedAnnotation.object_hash,
                ProjectTaggedAnnotation.tag_hash,
            ).where(
                ProjectTaggedAnnotation.project_hash == project_hash,
                in_op(
                    tuple_(
                        ProjectTaggedAnnotation.du_hash,
                        ProjectTaggedAnnotation.frame,
                        ProjectTaggedAnnotation.object_hash,
                    ),
                    label_selectors,
                ),
            )
        ).all()
        existing_label_tags: dict[tuple[uuid.UUID, int, str], set[uuid.UUID]] = {}
        for du_hash, frame, object_hash, tag_hash in existing_label_tag_tuples:
            existing_label_tags.setdefault((du_hash, frame, object_hash), set()).add(tag_hash)

        data_exists = set()
        label_exists = set()

        data_tags_to_add: list[ProjectTaggedDataUnit] = []
        data_tags_to_remove: set[tuple[uuid.UUID, int, uuid.UUID]] = set()  # du_hash, frame, tag_hash
        annotation_tags_to_add: list[ProjectTaggedAnnotation] = []
        annotation_tags_to_remove: set[
            tuple[uuid.UUID, int, str, uuid.UUID]
        ] = set()  # du_hash, frame, object_hash, tag_hash

        for item in payload:
            data_tag_list = item.grouped_tags["data"]
            annotation_tag_list = item.grouped_tags["label"]
            _, du_hash_str, frame_str, *annotation_hashes = item.id.split("_")
            du_hash = uuid.UUID(du_hash_str)
            frame = int(frame_str)
            new_data_tag_uuids: set[UUID] = set()
            for data_tag in data_tag_list:
                tag_hash = _get_or_create_tag_hash(data_tag)
                dup_key = (project_hash, du_hash, frame, tag_hash)
                if dup_key in data_exists:
                    continue
                data_exists.add(dup_key)
                new_data_tag_uuids.add(tag_hash)
                if tag_hash in existing_data_tags.get((du_hash, frame), set()):
                    continue
                data_tags_to_add.append(
                    ProjectTaggedDataUnit(
                        project_hash=project_hash,
                        du_hash=du_hash,
                        frame=frame,
                        tag_hash=tag_hash,
                    )
                )
            data_tags_to_remove.update(
                set(
                    [
                        (du_hash, frame, tag_hash)
                        for tag_hash in existing_data_tags.get((du_hash, frame), set()).difference(new_data_tag_uuids)
                    ]
                )
            )

            for annotation_hash in annotation_hashes:
                new_label_tag_uuids: set[UUID] = set()
                for annotation_tag in annotation_tag_list:
                    tag_hash = _get_or_create_tag_hash(annotation_tag)
                    dup_key2 = (project_hash, du_hash, frame, annotation_hash, tag_hash)
                    if dup_key2 in label_exists:
                        continue
                    new_label_tag_uuids.add(tag_hash)
                    label_exists.add(dup_key2)
                    if tag_hash in existing_label_tags.get((du_hash, frame, annotation_hash), set()):
                        continue
                    annotation_tags_to_add.append(
                        ProjectTaggedAnnotation(
                            project_hash=project_hash,
                            du_hash=du_hash,
                            frame=frame,
                            tag_hash=tag_hash,
                            object_hash=annotation_hash,
                        )
                    )

                annotation_tags_to_remove.update(
                    set(
                        [
                            (du_hash, frame, annotation_hash, tag_hash)
                            for tag_hash in existing_label_tags.get(
                                (du_hash, frame, annotation_hash), set()
                            ).difference(new_label_tag_uuids)
                        ]
                    )
                )

        # Delete left over data and annotation tags
        if data_tags_to_add:
            sess.add_all(data_tags_to_add)
        if annotation_tags_to_add:
            sess.add_all(annotation_tags_to_add)
        if data_tags_to_remove:
            sess.execute(
                delete(ProjectTaggedDataUnit).where(
                    ProjectTaggedDataUnit.project_hash == project_hash,
                    in_op(
                        tuple_(
                            ProjectTaggedDataUnit.du_hash, ProjectTaggedDataUnit.frame, ProjectTaggedDataUnit.tag_hash
                        ),
                        data_tags_to_remove,
                    ),
                )
            )

        if annotation_tags_to_remove:
            sess.execute(
                delete(ProjectTaggedAnnotation).where(
                    ProjectTaggedAnnotation.project_hash == project_hash,
                    in_op(
                        tuple_(
                            ProjectTaggedAnnotation.du_hash,
                            ProjectTaggedAnnotation.frame,
                            ProjectTaggedAnnotation.object_hash,
                            ProjectTaggedAnnotation.tag_hash,
                        ),
                        annotation_tags_to_remove,
                    ),
                )
            )
        sess.commit()


@router.get("/{project}/has_similarity_search")
def get_has_similarity_search(project: ProjectFileStructureDep, embedding_type: EmbeddingType):
    finder = get_similarity_finder(embedding_type, project)
    return finder.index_available


@router.get("/{project}/similarities/{id}")
def get_similar_items(
    project: ProjectFileStructureDep, id: str, embedding_type: EmbeddingType, page_size: Optional[int] = None
):
    finder = get_similarity_finder(embedding_type, project)
    if embedding_type == EmbeddingType.IMAGE:
        id = "_".join(id.split("_", maxsplit=3)[:3])
    return finder.get_similarities(id)


@router.get("/{project}/metrics")
def get_available_metrics(
    project: ProjectFileStructureDep,
    scope: Optional[MetricScope] = None,
    prediction_type: Optional[MainPredictionType] = None,
    prediction_outcome: Optional[Union[ClassificationOutcomeType, ObjectDetectionOutcomeType]] = None,
):
    if scope == MetricScope.PREDICTION:
        if prediction_type is None:
            raise ValueError("Prediction metrics requires prediction type")
        prediction_metrics, label_metrics, *_ = read_prediction_files(project, prediction_type)
        metrics = (
            label_metrics if prediction_outcome == ObjectDetectionOutcomeType.FALSE_NEGATIVES else prediction_metrics
        )
    else:
        metrics = load_project_metrics(project)

    results: Dict[MetricScope, List[dict]] = {
        MetricScope.DATA: [],
        MetricScope.ANNOTATION: [],
        MetricScope.PREDICTION: [],
    }
    for metric in natsorted(filter(filter_none_empty_metrics, metrics), key=lambda metric: metric.name):
        prediction_metric = "predictions" in metric.path.as_posix()
        label_outcome = prediction_outcome == ObjectDetectionOutcomeType.FALSE_NEGATIVES
        if metric.name in ["Object Count", "Frame object density"]:
            if (label_outcome and prediction_metric) or (not label_outcome and not prediction_metric):
                continue

        metric_result = {
            "name": metric.name,
            "embeddingType": get_embedding_type(metric.meta.annotation_type),
            "range": Range(min=metric.meta.stats.min_value, max=metric.meta.stats.max_value),
        }
        if metric.level == "F":
            results[MetricScope.DATA].append(metric_result)
        elif scope == MetricScope.PREDICTION:
            results[MetricScope.PREDICTION].append(metric_result)
        else:
            results[MetricScope.ANNOTATION].append(metric_result)

    return results


@router.get("/{project}/prediction_types")
def get_available_prediction_types(project: ProjectFileStructureDep):
    if check_model_prediction_availability(project.predictions):
        return [
            MainPredictionType.OBJECT
            if (project.predictions / "ground_truths_matched.json").exists()
            else MainPredictionType.CLASSIFICATION
        ]

    return [
        prediction_type
        for prediction_type in MainPredictionType
        if check_model_prediction_availability(project.predictions / prediction_type.value)
    ]


@router.post("/{project}/2d_embeddings", response_class=ORJSONResponse)
def get_2d_embeddings(
    project: ProjectFileStructureDep, embedding_type: Annotated[EmbeddingType, Body()], filters: Filters
):
    embeddings_df = get_2d_embedding_data(project, embedding_type)

    if embeddings_df is None:
        raise HTTPException(
            status_code=404, detail=f'Embeddings of type "{embedding_type}" were not found for project "{project}"'
        )

    filtered = filtered_merged_metrics(project, filters)

    embeddings_df.set_index("identifier", inplace=True)
    embeddings_df = cast(DataFrame[Embedding2DSchema], embeddings_df[embeddings_df.index.isin(filtered.index)])

    if filters.prediction_filters is not None:
        embeddings_df["data_row_id"] = partial_column(embeddings_df.index, 3)
        predictions, labels = get_model_predictions(project, PredictionsFilters(type=filters.prediction_filters.type))

        if filters.prediction_filters.type == MainPredictionType.OBJECT:
            labels = labels[[LabelMatchSchema.is_false_negative]]
            labels = labels[labels[LabelMatchSchema.is_false_negative]].copy()
            labels["data_row_id"] = partial_column(labels.index, 3)
            labels["score"] = 0
            labels.drop(LabelMatchSchema.is_false_negative, axis=1, inplace=True)
            predictions = predictions[[PredictionMatchSchema.is_true_positive]].copy()
            predictions["data_row_id"] = partial_column(predictions.index, 3)
            predictions.rename(columns={PredictionMatchSchema.is_true_positive: "score"}, inplace=True)

            merged_score = pd.concat([labels, predictions], axis=0)
            grouped_score = (
                merged_score.groupby("data_row_id")[Embedding2DScoreSchema.score].mean().to_frame().reset_index()
            )
            embeddings_df = cast(
                DataFrame[Embedding2DSchema],
                embeddings_df.merge(grouped_score, on="data_row_id", how="outer")
                .fillna(0)
                .rename({"data_row_id": "identifier"}, axis=1),
            )
        else:
            predictions = predictions[[ClassificationPredictionMatchSchema.is_true_positive]]
            predictions["data_row_id"] = partial_column(predictions.index, 3)

            embeddings_df = cast(
                DataFrame[Embedding2DSchema],
                embeddings_df.merge(predictions, on="data_row_id", how="outer")
                .drop(columns=[Embedding2DSchema.label])
                .rename(
                    columns={
                        ClassificationPredictionMatchSchema.is_true_positive: Embedding2DSchema.label,
                        "data_row_id": "identifier",
                    }
                ),
            )

            embeddings_df["score"] = embeddings_df[Embedding2DSchema.label]
            embeddings_df[Embedding2DSchema.label] = embeddings_df[Embedding2DSchema.label].apply(
                lambda x: "Correct Classification" if x == 1.0 else "Misclassification"
            )

    return ORJSONResponse(embeddings_df.reset_index().rename({"identifier": "id"}, axis=1).to_dict("records"))


@cached(cache=LRUCache(maxsize=10))
def get_querier(project: ProjectFileStructure):
    settings = get_settings()
    if settings.DEPLOYMENT_NAME is not None:
        project_dir = project.project_dir
        new_root = project_dir.parent / settings.DEPLOYMENT_NAME / project_dir.name
        project = ProjectFileStructure(new_root)
    return Querier(project)


class SearchType(str, Enum):
    SEARCH = "search"
    CODEGEN = "codegen"


def get_ids(ids_column: IndexOrSeries, scope: Optional[MetricScope] = None):
    if scope == MetricScope.DATA or scope == MetricScope.PREDICTION:
        return partial_column(ids_column, 3).unique().tolist()
    elif scope == MetricScope.ANNOTATION:
        data_ids = partial_column(ids_column, 3).unique().tolist()
        return ids_column[~ids_column.isin(data_ids)].tolist()
    return ids_column.tolist()


@router.post("/{project}/search", dependencies=[Depends(verify_premium)])
def search(
    project: ProjectFileStructureDep,
    type: Annotated[SearchType, Form()],
    filters: Annotated[str, Form()] = "",
    query: Annotated[Optional[str], Form()] = None,
    image: Annotated[Optional[UploadFile], File()] = None,
    scope: Annotated[Optional[MetricScope], Form()] = None,
):
    if not (query or (image is not None)):
        raise HTTPException(status_code=422, detail="Invalid query. Either `query` or `image` should be specified")

    if filters:
        _filters = Filters.parse_raw(filters)
    else:
        _filters = Filters()

    querier = get_querier(project)

    merged_metrics = filtered_merged_metrics(project, _filters)

    def _search(ids: List[str]):
        snippet = None
        if type == SearchType.SEARCH:
            image_bytes = None
            if image is not None:
                image_bytes = image.file.read()
            _query = CLIPQuery(text=query, image=image_bytes, limit=-1, identifiers=ids)
            result = querier.search_semantics(_query)
        else:
            text_query = TextQuery(text=query, limit=-1, identifiers=ids)
            result = querier.search_with_code(text_query)
            if result:
                snippet = result.snippet

        if not result:
            raise HTTPException(status_code=422, detail="Invalid query")

        return [item.identifier for item in result.result_identifiers], snippet

    if scope == MetricScope.PREDICTION and _filters.prediction_filters is not None:
        _, _, predictions, _ = read_prediction_files(project, _filters.prediction_filters.type)
        if predictions is not None:
            ids, snippet = _search(get_ids(predictions["identifier"], scope))
            prediction_ids = predictions["identifier"].sort_values(
                key=lambda column: partial_column(column, 3).map(lambda id: ids.index(id))
            )
            return {"ids": prediction_ids.to_list(), "snippet": snippet}

    ids, snippet = _search(get_ids(merged_metrics.index, scope))
    return {"ids": ids, "snippet": snippet}


class CreateSubsetJSON(BaseModel):
    filters: Filters
    ids: List[str]
    project_title: str
    dataset_title: str
    project_description: Optional[str] = None
    dataset_description: Optional[str] = None


@router.post("/{project}/create_subset")
def create_subset(curr_project_structure: ProjectFileStructureDep, item: CreateSubsetJSON):
    if get_settings().ENV == "sandbox":
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Subsetting is not allowed in the current environment"
        )
    project_title = item.project_title
    project_description = item.project_description
    dataset_title = item.dataset_title
    dataset_description = item.dataset_description
    filtered_df = filtered_merged_metrics(curr_project_structure, item.filters).reset_index()
    if len(item.ids):
        filtered_df = filtered_df[filtered_df["identifier"].isin(item.ids)]
    target_project_dir = curr_project_structure.project_dir.parent / project_title.lower().replace(" ", "-")
    target_project_structure = ProjectFileStructure(target_project_dir)
    current_project_meta = curr_project_structure.load_project_meta()
    remote_copy = current_project_meta.get("has_remote", False)

    if target_project_dir.exists():
        raise Exception("Subset with the same title already exists")
    target_project_dir.mkdir()

    try:
        ids_df = filtered_df["identifier"].str.split("_", n=4, expand=True)
        filtered_lr_du = {LabelRowDataUnit(label_row, data_unit) for label_row, data_unit in zip(ids_df[0], ids_df[1])}
        filtered_label_rows = {lr_du.label_row for lr_du in filtered_lr_du}
        filtered_data_hashes = {lr_du.data_unit for lr_du in filtered_lr_du}
        filtered_labels = {(ids[1][0], ids[1][1], ids[1][3] if len(ids[1]) > 3 else None) for ids in ids_df.iterrows()}

        create_filtered_db(target_project_dir, filtered_df)

        if curr_project_structure.image_data_unit.exists():
            copy_image_data_unit_json(curr_project_structure, target_project_structure, filtered_data_hashes)

        filtered_label_row_meta = copy_label_row_meta_json(
            curr_project_structure, target_project_structure, filtered_label_rows
        )

        label_rows = {label_row for label_row in filtered_label_row_meta.keys()}

        shutil.copy2(curr_project_structure.ontology, target_project_structure.ontology)

        copy_project_meta(
            curr_project_structure,
            target_project_structure,
            project_title,
            project_description or "",
            final_data_version=202306141750,
        )

        create_filtered_metrics(curr_project_structure, target_project_structure, filtered_df)

        # Only run data-migrations up-to just before global db migration script
        ensure_safe_project(target_project_structure.project_dir, final_data_version=202306141750)
        copy_filtered_data(
            curr_project_structure,
            target_project_structure,
            filtered_label_rows,
            filtered_data_hashes,
            filtered_labels,
        )

        create_filtered_embeddings(
            curr_project_structure, target_project_structure, filtered_label_rows, filtered_data_hashes, filtered_df
        )

        if remote_copy:
            original_project = get_encord_project(
                current_project_meta["ssh_key_path"], current_project_meta["project_hash"]
            )
            dataset_hash_map: dict[str, set[str]] = {}
            for k, v in filtered_label_row_meta.items():
                dataset_hash_map.setdefault(v["dataset_hash"], set()).add(v["data_hash"])

            cloned_project_hash = original_project.copy_project(
                new_title=project_title,
                new_description=project_description,
                copy_collaborators=True,
                copy_datasets=CopyDatasetOptions(
                    action=CopyDatasetAction.CLONE,
                    dataset_title=dataset_title,
                    dataset_description=dataset_description,
                    datasets_to_data_hashes_map={k: list(v) for k, v in dataset_hash_map.items()},
                ),
                copy_labels=CopyLabelsOptions(
                    accepted_label_statuses=[state for state in ReviewApprovalState],
                    accepted_label_hashes=list(label_rows),
                ),
            )
            cloned_project = get_encord_project(current_project_meta["ssh_key_path"], cloned_project_hash)
            cloned_project_label_rows = [
                cloned_project.get_label_row(src_row.label_hash) for src_row in cloned_project.list_label_rows_v2()
            ]
            filtered_du_lr_mapping = {lrdu.data_unit: lrdu.label_row for lrdu in filtered_lr_du}

            def _get_one_data_unit(lr: dict, valid_data_units: dict) -> str:
                data_units = lr["data_units"]
                for data_unit_key in data_units.keys():
                    if data_unit_key in valid_data_units:
                        return data_unit_key
                raise StopIteration(
                    f"Cannot find data unit to lookup: {list(data_units.keys())}, {list(valid_data_units.keys())}"
                )

            lr_du_mapping = {
                # We only use the label hash as the key for database migration. The data hashes are preserved anyway.
                LabelRowDataUnit(
                    filtered_du_lr_mapping[_get_one_data_unit(lr, filtered_du_lr_mapping)],
                    lr["data_hash"],  # This value is the same
                ): LabelRowDataUnit(lr["label_hash"], lr["data_hash"])
                for lr in cloned_project_label_rows
            }

            with PrismaConnection(target_project_structure) as conn:
                original_label_rows = conn.labelrow.find_many()
            original_label_row_map = {
                original_label_row.label_hash: json.loads(original_label_row.label_row_json or "")
                for original_label_row in original_label_rows
            }

            new_label_row_map = {label_row["label_hash"]: label_row for label_row in cloned_project_label_rows}

            label_row_json_map = {}
            for (old_lr, old_du), (new_lr, new_du) in lr_du_mapping.items():
                lr = dict(original_label_row_map[old_lr])
                lr["label_hash"] = new_label_row_map[new_lr]["label_hash"]
                lr["dataset_hash"] = new_label_row_map[new_lr]["dataset_hash"]
                label_row_json_map[new_lr] = json.dumps(lr)

            project_meta = fetch_project_meta(target_project_structure.project_dir)
            project_meta["has_remote"] = True
            project_meta["project_hash"] = cloned_project_hash
            update_project_meta(target_project_structure.project_dir, project_meta)

            du_hash_map = DataHashMapping()

            replace_uids(
                target_project_structure,
                lr_du_mapping,
                du_hash_map,
                original_project.project_hash,
                cloned_project_hash,
                cloned_project.datasets[0]["dataset_hash"],
            )

            # Sync database identifiers
            replace_db_uids(
                target_project_structure,
                du_hash_map=DataHashMapping(),  # Preserved and used as migration key
                lr_du_mapping=lr_du_mapping,  # Update label hash and lr_dr hashes ( label hash)
                label_row_json_map=label_row_json_map,  # Update label row jsons to correct value.
            )
        else:
            # Replace all label hashes with different values, to bypass the label_hash unique constraint bug
            # this will regenerate a unique label hash and dataset hash for the subset project.
            new_project_hash = target_project_structure.load_project_meta()["project_hash"]
            dataset_hash = str(uuid.uuid4())
            with PrismaConnection(target_project_structure) as prisma_conn:
                prisma_label_rows = prisma_conn.labelrow.find_many()
                lh_map: Dict[str, str] = {
                    label_row.label_hash or "": str(uuid.uuid4()) for label_row in prisma_label_rows
                }
                lr_du_mapping = {
                    LabelRowDataUnit(label_row.label_hash or "", label_row.data_hash): LabelRowDataUnit(
                        lh_map[label_row.label_hash or ""], label_row.data_hash
                    )
                    for label_row in prisma_label_rows
                }
                label_row_json_map_2: Dict[str, dict] = {
                    lh_map[label_row.label_hash or ""]: json.loads(label_row.label_row_json or "")
                    for label_row in prisma_label_rows
                }
                for label_hash, label_row_json in label_row_json_map_2.items():
                    label_row_json["dataset_hash"] = dataset_hash
                    label_row_json["label_hash"] = label_hash
            replace_uids(
                target_project_structure,
                lr_du_mapping,
                DataHashMapping(),
                new_project_hash,
                new_project_hash,
                dataset_hash,
            )
            replace_db_uids(
                target_project_structure,
                du_hash_map=DataHashMapping(),
                lr_du_mapping=lr_du_mapping,
                label_row_json_map={k: json.dumps(v) for k, v in label_row_json_map_2.items()},
                refresh=False,  # Not a remote project running the migration
            )

    except Exception as e:
        shutil.rmtree(target_project_dir.as_posix())
        raise e

    # On Success:
    # Mirror to global sqlite database
    # migrate_disk_to_db(target_project_structure)
    # run all migration scripts
    # FIXME: hacky
    ensure_safe_project(target_project_structure.project_dir)

    # Project now exists - invalidate cache
    get_all_projects.cache_clear()  # type: ignore


class UploadToEncordModel(BaseModel):
    dataset_title: str
    dataset_description: str
    project_title: str
    project_description: str
    ontology_title: Optional[str]
    ontology_description: str


@router.post("/{project}/upload_to_encord")
def upload_to_encord(
    pfs: ProjectFileStructureDep,
    item: UploadToEncordModel,
):
    if get_settings().ENV in ["sandbox", "production"]:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Uploading is not allowed in the current environment"
        )

    old_project_meta = pfs.load_project_meta()
    old_project_hash = uuid.UUID(old_project_meta["project_hash"])
    old_project_name = str(old_project_meta["project_title"])
    with DBConnection(pfs) as conn:
        df = MergedMetrics(conn).all()
    encord_actions = EncordActions(pfs.project_dir, app_config.get_ssh_key())
    try:
        dataset_creation_result = encord_actions.create_dataset(
            dataset_title=item.dataset_title, dataset_description=item.dataset_description, dataset_df=df
        )
    except DatasetUniquenessError as e:
        return None

    # FIXME: existing logic re-uses 'ontology_hash'
    ontology_hash = encord_actions.create_ontology(
        title=item.ontology_title or item.project_title, description=item.ontology_description or ""
    ).ontology_hash

    try:
        new_project = encord_actions.create_project(
            dataset_creation_result=dataset_creation_result,
            project_title=item.project_title,
            project_description=item.project_description,
            ontology_hash=ontology_hash,
        )

        # Regenerate new database instance
        pfs.cache_clear()
        new_project_meta = pfs.load_project_meta()
        new_project_hash = uuid.UUID(new_project_meta["project_hash"])
        if new_project_hash == old_project_hash:
            raise ValueError("BUG: Upload to encord hasn't changed project hash")

        # Move folder so uuid lookup will work correctly.
        migrate_disk_to_db(pfs)
        delete_project_from_db(engine, old_project_hash)
        with Session(engine) as sess:
            # The project name has to be reverted to the same value
            # Update some metadata
            new_db = sess.exec(select(Project).where(Project.project_hash == new_project_hash)).first()
            if new_db is None:
                raise ValueError("Missing new project in the database when uploading")
            new_db.project_name = old_project_name
            sess.add(new_db)
            sess.commit()
    except Exception as e:
        print(str(e))
        raise e

    # Project now exists - invalidate cache
    get_all_projects.cache_clear()  # type: ignore
    return {
        "project_hash": new_project.project_hash,
        "dataset_hash": dataset_creation_result.hash,
    }


def _download_task(pfs: ProjectFileStructure, project_name: str):
    pfs.project_dir.mkdir(exist_ok=True)
    project_dir = fetch_prebuilt_project(project_name, pfs.project_dir)
    ensure_safe_project(project_dir)
    migrate_disk_to_db(pfs)


@router.get("/{project}/download_sandbox")
def download_sandbox_project(project: str, background_tasks: BackgroundTasks):
    sandbox_projects = available_prebuilt_projects(get_settings().AVAILABLE_SANDBOX_PROJECTS)
    sandbox_project = next(
        (sandbox_project for sandbox_project in sandbox_projects.values() if sandbox_project["hash"] == project), None
    )
    if not sandbox_project:
        raise HTTPException(status_code=404, detail=f'Sandbox project with hash "{project}" was not found')

    pfs = ProjectFileStructure(get_settings().SERVER_START_PATH / sandbox_project["name"])

    with Session(engine) as sess:
        sess.add(
            Project(
                project_hash=UUID(project),
                project_name="temp",
                project_description="",
                project_remote_ssh_key_path=None,
                project_ontology={},
            )
        )
        sess.commit()

    background_tasks.add_task(_download_task, pfs, sandbox_project["name"])
    return "Downloading in background"
