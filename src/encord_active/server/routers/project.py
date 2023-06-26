import re
from enum import Enum
from functools import lru_cache
from typing import Annotated, List, Optional, Union

import pandas as pd
from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import ORJSONResponse
from natsort import natsorted
from pydantic import BaseModel

from encord_active.app.model_quality.prediction_types.lib_object_type_builder import (
    ClassificationOutcomeType,
    ObjectDetectionOutcomeType,
    PredictionsFilters,
    get_model_prediction_by_id,
    get_model_predictions,
    read_prediction_files,
)
from encord_active.lib.common.filtering import Filters
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.helpers.tags import (
    GroupedTags,
    Tag,
    all_tags,
    from_grouped_tags,
    to_grouped_tags,
)
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.embeddings.dimensionality_reduction import get_2d_embedding_data
from encord_active.lib.embeddings.types import Embedding2DSchema, Embedding2DScoreSchema
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.metrics.utils import (
    MetricScope,
    filter_none_empty_metrics,
    get_embedding_type,
)
from encord_active.lib.model_predictions.reader import (
    ClassificationPredictionMatchSchema,
    LabelMatchSchema,
    PredictionMatchSchema,
)
from encord_active.lib.model_predictions.writer import MainPredictionType
from encord_active.lib.premium.model import TextQuery
from encord_active.lib.premium.querier import Querier
from encord_active.lib.project.project_file_structure import ProjectFileStructure
from encord_active.server.dependencies import (
    ProjectFileStructureDep,
    verify_premium,
    verify_token,
)
from encord_active.server.settings import get_settings
from encord_active.server.utils import (
    filtered_merged_metrics,
    get_similarity_finder,
    load_project_metrics,
    partial_column,
    to_item,
)

router = APIRouter(
    prefix="/projects",
    tags=["projects"],
    dependencies=[Depends(verify_token)],
)


@router.post("/{project}/item_ids_by_metric", response_class=ORJSONResponse)
def read_item_ids(
    project: ProjectFileStructureDep,
    sort_by_metric: Annotated[str, Body()],
    filters: Filters = Filters(),
    ascending: Annotated[bool, Body()] = True,
    ids: Annotated[Optional[list[str]], Body()] = None,
):
    merged_metrics = filtered_merged_metrics(project, filters)

    try:
        column = [col for col in merged_metrics.columns if col.lower() == sort_by_metric.lower()][0]
        df = merged_metrics
    except:
        try:
            if filters.prediction_filters is None:
                raise
            df, _ = get_model_predictions(project, filters.prediction_filters)

            if filters.prediction_filters.outcome == ObjectDetectionOutcomeType.FALSE_NEGATIVES:
                sort_by_metric = re.sub(r"(.*?\()P(\))", r"\1O\2", sort_by_metric)

            column = [col for col in df.columns if col.lower() == sort_by_metric.lower()][0]
            df = df[partial_column(df.index, 3).isin(partial_column(merged_metrics.index, 3).unique())]
            if filters.prediction_filters.outcome == ObjectDetectionOutcomeType.FALSE_NEGATIVES:
                df = df[df.index.isin(merged_metrics.index)]
        except:
            raise Exception("Couldn't find the selected metric in the the project")

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
    with DBConnection(project) as conn:
        df = MergedMetrics(conn).all(columns=["tags"]).reset_index()
    records = df[df["tags"].str.len() > 0].to_dict("records")

    # Assign the respective frame's data tags to the rows representing the labels
    data_row_id_to_data_tags: dict[str, List[Tag]] = dict()
    for row in records:
        data_row_id = "_".join(row["identifier"].split("_", maxsplit=3)[:3])
        if row["identifier"] == data_row_id:  # The row contains the info related to a frame
            data_row_id_to_data_tags[data_row_id] = row.get("tags", [])
    for row in records:
        data_row_id = "_".join(row["identifier"].split("_", maxsplit=3)[:3])
        if row["identifier"] != data_row_id:  # The row contains the info related to some labels
            selected_tags = row.setdefault("tags", [])
            selected_tags.extend(data_row_id_to_data_tags.get(data_row_id, []))

    return {record["identifier"]: to_grouped_tags(record["tags"]) for record in records}


@router.get("/{project}/items/{id:path}")
def read_item(project: ProjectFileStructureDep, id: str):
    lr_hash, du_hash, frame, *object_hash = id.split("_")

    row = get_model_prediction_by_id(project, id)

    if row:
        return to_item(row, project, lr_hash, du_hash, frame, object_hash[0] if len(object_hash) else None)

    with DBConnection(project) as conn:
        row = MergedMetrics(conn).get_row(id).dropna(axis=1).to_dict("records")[0]

        # Include data tags from the relevant frame when the inspected item is a label
        data_row_id = "_".join(row["identifier"].split("_", maxsplit=3)[:3])
        if row["identifier"] != data_row_id:
            data_row = MergedMetrics(conn).get_row(data_row_id).dropna(axis=1).to_dict("records")[0]
            selected_tags = row.setdefault("tags", [])
            selected_tags.extend(data_row.get("tags", []))

    return to_item(row, project, lr_hash, du_hash, frame)


class ItemTags(BaseModel):
    id: str
    grouped_tags: GroupedTags


@router.put("/{project}/item_tags")
def tag_items(project: ProjectFileStructureDep, payload: List[ItemTags]):
    with DBConnection(project) as conn:
        for item in payload:
            data_tags, label_tags = from_grouped_tags(item.grouped_tags)
            data_row_id = "_".join(item.id.split("_", maxsplit=3)[:3])

            # Update the data tags associated with the frame (or the one that contains the labels)
            MergedMetrics(conn).update_tags(data_row_id, data_tags)

            # Update the label tags associated with the labels (if they exist)
            if item.id != data_row_id:
                MergedMetrics(conn).update_tags(item.id, label_tags)


@router.get("/{project}/has_similarity_search")
def get_has_similarity_search(project: ProjectFileStructureDep, embedding_type: EmbeddingType):
    finder = get_similarity_finder(embedding_type, project)
    return finder.index_available


@router.get("/{project}/similarities/{id}")
def get_similar_items(
    project: ProjectFileStructureDep, id: str, embedding_type: EmbeddingType, page_size: Optional[int] = None
):
    finder = get_similarity_finder(embedding_type, project)
    return finder.get_similarities(id)


@router.get("/{project}/metrics")
def get_available_metrics(
    project: ProjectFileStructureDep,
    scope: Optional[MetricScope] = None,
    prediction_type: Optional[MainPredictionType] = None,
    prediction_outcome: Optional[Union[ClassificationOutcomeType, ObjectDetectionOutcomeType]] = None,
):
    if scope == MetricScope.MODEL_QUALITY:
        prediction_metrics, label_metrics, *_ = read_prediction_files(project, prediction_type)
        metrics = (
            label_metrics if prediction_outcome == ObjectDetectionOutcomeType.FALSE_NEGATIVES else prediction_metrics
        )
    else:
        metrics = load_project_metrics(project, scope)
    return [
        {"name": metric.name, "embeddingType": get_embedding_type(metric.meta.annotation_type)}
        for metric in natsorted(filter(filter_none_empty_metrics, metrics), key=lambda metric: metric.name)
    ]


@router.post("/{project}/2d_embeddings", response_class=ORJSONResponse)
def get_2d_embeddings(
    project: ProjectFileStructureDep, embedding_type: Annotated[EmbeddingType, Body()], filters: Filters
):
    embeddings_df = get_2d_embedding_data(project, embedding_type)
    embeddings_df.set_index("identifier", inplace=True)

    if embeddings_df is None:
        raise HTTPException(
            status_code=404, detail=f"Embeddings of type: {embedding_type} were not found for project: {project}"
        )

    filtered = filtered_merged_metrics(project, filters)
    embeddings_df = embeddings_df[embeddings_df.index.isin(filtered.index)]

    if filters.prediction_filters is not None:
        embeddings_df["data_row_id"] = partial_column(embeddings_df.index, 3)
        predictions, labels = get_model_predictions(project, PredictionsFilters(type=filters.prediction_filters.type))

        if filters.prediction_filters.type == MainPredictionType.OBJECT:
            labels = labels[[LabelMatchSchema.is_false_negative]]
            labels = labels[labels[LabelMatchSchema.is_false_negative] == True]
            labels["data_row_id"] = partial_column(labels.index, 3)
            labels["score"] = 0
            labels.drop(LabelMatchSchema.is_false_negative, axis=1, inplace=True)
            predictions = predictions[[PredictionMatchSchema.is_true_positive]]
            predictions["data_row_id"] = partial_column(predictions.index, 3)
            predictions.rename(columns={PredictionMatchSchema.is_true_positive: "score"}, inplace=True)

            merged_score = pd.concat([labels, predictions], axis=0)
            grouped_score = (
                merged_score.groupby("data_row_id")[Embedding2DScoreSchema.score].mean().to_frame().reset_index()
            )
            embeddings_df = embeddings_df.merge(grouped_score, on="data_row_id", how="outer").fillna(0)
            embeddings_df.rename({"data_row_id": "identifier"}, axis=1, inplace=True)
        else:
            predictions = predictions[[ClassificationPredictionMatchSchema.is_true_positive]]
            predictions["data_row_id"] = partial_column(predictions.index, 3)

            embeddings_df = embeddings_df.merge(predictions, on="data_row_id", how="outer")

            embeddings_df.drop(columns=[Embedding2DSchema.label], inplace=True)
            embeddings_df.rename(
                columns={
                    ClassificationPredictionMatchSchema.is_true_positive: Embedding2DSchema.label,
                    "data_row_id": "identifier",
                },
                inplace=True,
            )

            embeddings_df[Embedding2DSchema.label] = embeddings_df[Embedding2DSchema.label].apply(
                lambda x: "True prediction" if x == 1.0 else "False prediction"
            )

    return ORJSONResponse(embeddings_df.reset_index().rename({"identifier": "id"}, axis=1).to_dict("records"))


@router.get("/{project}/tags")
def get_tags(project: ProjectFileStructureDep):
    return to_grouped_tags(all_tags(project))


@lru_cache
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


@router.get("/{project}/search", dependencies=[Depends(verify_premium)])
def search(project: ProjectFileStructureDep, query: str, type: SearchType, scope: Optional[MetricScope] = None):
    if not query:
        raise HTTPException(status_code=422, detail="Invalid query")
    querier = get_querier(project)

    with DBConnection(project) as conn:
        df = MergedMetrics(conn).all(False, columns=["identifier"])

    if scope == MetricScope.DATA_QUALITY:
        ids = df.index.str.split("_", n=3).str[0:3].str.join("_").unique().tolist()
    elif scope == MetricScope.LABEL_QUALITY:
        data_ids = df.index.str.split("_", n=3).str[0:3].str.join("_")
        ids = df.index[~df.index.isin(data_ids)].tolist()
    else:
        ids = df.index.tolist()

    snippet = None
    text_query = TextQuery(text=query, limit=-1, identifiers=ids)
    if type == SearchType.SEARCH:
        result = querier.search_semantics(text_query)
    else:
        result = querier.search_with_code(text_query)
        if result:
            snippet = result.snippet

    if not result:
        raise HTTPException(status_code=422, detail="Invalid query")

    return {
        "ids": [item.identifier for item in result.result_identifiers],
        "snippet": snippet,
    }
