from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, List, Optional
import pandas as pd
import numpy as np
from pandera.typing import DataFrame

from fastapi import APIRouter, Body, Depends, HTTPException
from fastapi.responses import ORJSONResponse
from natsort import natsorted
from pydantic import BaseModel

from encord_active.app.projects_page import get_projects
from encord_active.lib.charts.data_quality_summary import CrossMetricSchema
from encord_active.lib.dataset.outliers import get_all_metrics_outliers, AllMetricsOutlierSchema, \
    MetricWithDistanceSchema, _COLUMNS
from encord_active.lib.dataset.summary_utils import get_all_image_sizes, get_median_value_of_2d_array, \
    get_metric_summary
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
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.metrics.utils import (
    MetricScope,
    filter_none_empty_metrics,
    get_embedding_type,
)
from encord_active.lib.premium.model import TextQuery
from encord_active.lib.premium.querier import Querier
from encord_active.lib.project.metadata import fetch_project_meta
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
    to_item, to_preview_item,
)

router = APIRouter(
    prefix="/projects",
    tags=["projects"],
    dependencies=[Depends(verify_token)],
)


@router.get("/list_all_projects")
def list_all_projects():
    project_path = get_settings().SERVER_START_PATH
    res = get_projects(project_path)
    result = {}
    for k, v in res.projects.items():
        if v["path"] is not None:
            # (should take hash as argument not path)
            #  but currently all queries assume that the file path instead of the project is an argument
            #  so return the incorrect value for the time being.
            project_meta = fetch_project_meta(Path(v["path"]))
            project_id = Path(v["path"]).name
            result[project_id] = {
                "title": project_meta["project_title"],
                "description": project_meta["project_description"],
                "project_hash": project_meta["project_hash"],
            }
    return result


@router.post("/{project}/item_ids_by_metric", response_class=ORJSONResponse)
def read_item_ids(
    project: ProjectFileStructureDep,
    sort_by_metric: Annotated[str, Body()],
    filters: Filters = Filters(),
    ascending: Annotated[bool, Body()] = True,
    ids: Annotated[Optional[list[str]], Body()] = None,
):
    merged_metrics = filtered_merged_metrics(project, filters)
    column = [col for col in merged_metrics.columns if col.lower() == sort_by_metric.lower()][0]
    res = merged_metrics[[column]].dropna().sort_values(by=[column], ascending=ascending)
    if ids:
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
    lr_hash, du_hash, frame, *_ = id.split("_")
    with DBConnection(project) as conn:
        row = MergedMetrics(conn).get_row(id).dropna(axis=1).to_dict("records")[0]

        # Include data tags from the relevant frame when the inspected item is a label
        data_row_id = "_".join(row["identifier"].split("_", maxsplit=3)[:3])
        if row["identifier"] != data_row_id:
            data_row = MergedMetrics(conn).get_row(data_row_id).dropna(axis=1).to_dict("records")[0]
            selected_tags = row.setdefault("tags", [])
            selected_tags.extend(data_row.get("tags", []))

    return to_item(row, project, lr_hash, du_hash, frame)


@router.get("/{project}/item_preview/{data_hash}/{frame}/{object_hash}")
@router.get("/{project}/item_preview/{data_hash}/{frame}/")
def get_item_preview(project: ProjectFileStructureDep, data_hash: str, frame: int, object_hash: Optional[str] = None):
    return to_preview_item(project, data_hash, frame, object_hash)


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
def get_available_metrics(project: ProjectFileStructureDep, scope: Optional[MetricScope] = None):
    metrics = load_project_metrics(project, scope)
    return [
        {"name": metric.name, "embeddingType": get_embedding_type(metric.meta.annotation_type)}
        for metric in natsorted(filter(filter_none_empty_metrics, metrics), key=lambda metric: metric.name)
    ]


@router.get("/{project}/data_metrics_summary")
def get_metrics_summary(project: ProjectFileStructureDep):
    image_sizes = get_all_image_sizes(project)
    median_image_dimension = get_median_value_of_2d_array(image_sizes)
    metrics = load_project_metrics(project, None)
    metrics_data_summary = get_metric_summary(metrics)
    all_metrics_outliers = get_all_metrics_outliers(metrics_data_summary)

    return {
        "dataCount": len(image_sizes),
        "medianWidth": int(median_image_dimension[0]),
        "medianHeight": int(median_image_dimension[1]),
        "metrics": [
            {
                "ident": str(row[AllMetricsOutlierSchema.metric_name]),
                "title": str(row[AllMetricsOutlierSchema.metric_name]),
                "short_summary": "",
                "long_summary": "",
                "statistics": {
                    # FIXME: added these extra statistics
                    "min": 0,
                    "q1": 0,
                    "median": 0,
                    "q3": 0,
                    "max": 0,
                    "severeOutlierCount": int(row[AllMetricsOutlierSchema.total_severe_outliers]),
                    "moderateOutlierCount": int(row[AllMetricsOutlierSchema.total_moderate_outliers]),
                }
            }
            for counter, (_, row) in enumerate(all_metrics_outliers.iterrows())
        ]
    }


@router.get("/{project}/images_sizes")
def get_metrics_summary(project: ProjectFileStructureDep):
    image_sizes = get_all_image_sizes(project)
    return {
        "sampleFraction": 1.0,
        "samples": [
            {
                "w": int(entry[0]),
                "h": int(entry[1]),
                "count": 1,
            }
            for entry in image_sizes.tolist()
        ]
    }


@router.get("/{project}/images_outliers/{metric}")
def get_images_outliers(
        project: ProjectFileStructureDep,
        metric: str, iqr_dist: float = float("inf"),
        offset: Optional[int] = None, limit: Optional[int] = None):
    metrics = load_project_metrics(project, None)
    metrics_data_summary = get_metric_summary(metrics)
    metric_item = metrics_data_summary.metrics.get(metric, None)
    if metric_item is None:
        return {
            "maxIQR": 0.0,
            "samples": [],
        }
    iqr_outliers = metric_item.iqr_outliers

    if iqr_outliers.n_severe_outliers + iqr_outliers.n_moderate_outliers == 0:
        return {
            "maxIQR": 0.0,
            "outliers": [],
        }

    df = metric_item.df
    max_value = float(df[_COLUMNS.dist_to_iqr].max())

    # FIXME: min_value = float(df[_COLUMNS.dist_to_iqr].min())

    def make_id(identifier: str) -> dict:
        label_hash, data_hash, frame, *_ = identifier.split("_")
        return {
            "labelHash": label_hash,
            "dataHash": data_hash,
            "frame": frame,
        }

    selected_df: DataFrame[MetricWithDistanceSchema] = df[df[_COLUMNS.dist_to_iqr] <= iqr_dist].fillna('').rename(
        columns={"dist_to_iqr": "iqrDist"}
    ).drop(columns=["object_class", "url", "frame", "index"])
    total_samples = len(selected_df['iqrDist'])
    selected_df['id'] = selected_df['identifier'].map(make_id)
    selected_df = selected_df.drop(columns=["identifier"])
    if limit is not None:
        real_offset = offset or 0
        selected_df = selected_df[real_offset: real_offset + limit]
    elif offset is not None:
        selected_df = selected_df[offset:]
    return {
        "maxIQR": max_value,
        "totalSamples": total_samples,
        "outliers": selected_df.to_dict("records")
    }


@router.get("/{project}/metrics_comparison")
def get_images_outliers(project: ProjectFileStructureDep, x_metric_name: str, y_metric_name: str):
    metrics = load_project_metrics(project, None)
    metrics_data_summary = get_metric_summary(metrics)
    if x_metric_name not in metrics_data_summary.metrics or y_metric_name not in metrics_data_summary.metrics:
        return {
            "sampleFraction": 1.0,
            "samples": [],
            "m": 0,
            "c": 0,
        }

    x_metric_df = (
        metrics_data_summary.metrics[str(x_metric_name)]
        .df[[MetricWithDistanceSchema.identifier, MetricWithDistanceSchema.score]]
        .copy()
    )
    x_metric_df.rename(columns={MetricWithDistanceSchema.score: f"{CrossMetricSchema.x}"}, inplace=True)

    y_metric_df = (
        metrics_data_summary.metrics[str(y_metric_name)]
        .df[[MetricWithDistanceSchema.identifier, MetricWithDistanceSchema.score]]
        .copy()
    )
    y_metric_df.rename(columns={MetricWithDistanceSchema.score: f"{CrossMetricSchema.y}"}, inplace=True)

    if x_metric_df.shape[0] == 0 or y_metric_df.shape[0] == 0:
        return {
            "sampleFraction": 1.0,
            "samples": [],
            "m": 0,
            "c": 0,
        }

    if len(x_metric_df.iloc[0][MetricWithDistanceSchema.identifier].split("_")) == len(
            y_metric_df.iloc[0][MetricWithDistanceSchema.identifier].split("_")
    ):
        merged_metrics = pd.merge(x_metric_df, y_metric_df, how="inner", on=MetricWithDistanceSchema.identifier)
    else:
        x_changed, to_be_parsed_df = (
            (True, x_metric_df.copy(deep=True))
            if len(x_metric_df.iloc[0][MetricWithDistanceSchema.identifier].split("_")) == 4
            else (False, y_metric_df.copy(deep=True))
        )

        to_be_parsed_df[[MetricWithDistanceSchema.identifier, "identifier_rest"]] = to_be_parsed_df[
            MetricWithDistanceSchema.identifier
        ].str.rsplit("_", n=1, expand=True)

        merged_metrics = pd.merge(
            to_be_parsed_df if x_changed else x_metric_df,
            y_metric_df if x_changed else to_be_parsed_df,
            how="inner",
            on=MetricWithDistanceSchema.identifier,
        )
        merged_metrics[MetricWithDistanceSchema.identifier] = (
                merged_metrics[MetricWithDistanceSchema.identifier] + "_" + merged_metrics["identifier_rest"]
        )

        merged_metrics.pop("identifier_rest")
    if merged_metrics.empty:
        return {
            "sampleFraction": 1.0,
            "samples": [],
            "m": 0,
            "c": 0,
        }

    samples: DataFrame = merged_metrics.pipe(DataFrame[CrossMetricSchema])
    # Derive linear regression
    x = samples["x"].to_numpy()
    y = samples["y"].to_numpy()
    a = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(a, y, rcond=None)[0]
    return {
        "sampleFraction": 1.0,
        "samples": samples.to_dict("records"),
        "m": m,
        "c": c,
    }


@router.post("/{project}/2d_embeddings", response_class=ORJSONResponse)
def get_2d_embeddings(
    project: ProjectFileStructureDep, embedding_type: Annotated[EmbeddingType, Body()], filters: Filters
):
    embeddings_df = get_2d_embedding_data(project, embedding_type)

    if embeddings_df is None:
        raise HTTPException(
            status_code=404, detail=f"Embeddings of type: {embedding_type} were not found for project: {project}"
        )

    filtered = filtered_merged_metrics(project, filters)
    embeddings_df[embeddings_df["identifier"].isin(filtered.index)]

    return ORJSONResponse(embeddings_df.rename({"identifier": "id"}, axis=1).to_dict("records"))


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
