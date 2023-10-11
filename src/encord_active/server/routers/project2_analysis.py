import functools
import io
import uuid
from typing import Annotated, Dict, List, Literal, Optional, Tuple

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from PIL import Image, ImageOps
from pydantic import BaseModel
from scipy.stats import ks_2samp
from sqlalchemy.engine import Engine
from sqlalchemy.sql.operators import is_not
from sqlmodel import Session, select

from encord_active.lib.embeddings.models.clip_embedder import CLIPEmbedder
from encord_active.server.dependencies import (
    DataOrAnnotateItem,
    dep_engine_readonly,
    parse_optional_data_or_annotate_item,
)
from encord_active.server.routers.queries import (
    metric_query,
    search_query,
    similarity_query,
)
from encord_active.server.routers.queries.domain_query import (
    TABLES_ANNOTATION,
    TABLES_DATA,
    DomainTables,
    Tables,
)
from encord_active.server.routers.queries.metric_query import literal_bucket_depends
from encord_active.server.routers.queries.search_query import (
    SearchFiltersFastAPIDepends,
)
from encord_active.server.routers.route_tags import AnalysisDomain, RouteTag

router = APIRouter(
    prefix="/{project_hash}/analysis/{domain}",
    tags=[RouteTag.PROJECT],
)

MODERATE_IQR_SCALE = 1.5
SEVERE_IQR_SCALE = 2.5


def _get_metric_domain_tables(domain: AnalysisDomain) -> Tables:
    if domain == AnalysisDomain.Data:
        return TABLES_DATA
    else:
        return TABLES_ANNOTATION


def _pack_id(du_hash: uuid.UUID, frame: int, annotation_hash: Optional[str] = None) -> str:
    if annotation_hash is None:
        return f"{du_hash}_{frame}"
    else:
        return f"{du_hash}_{frame}_{annotation_hash}"


def _unpack_id(ident: str) -> Tuple[uuid.UUID, int, Optional[str]]:
    values = ident.split("_")
    if len(values) == 2:
        du_hash_str, frame_str = values
        annotation_hash = None
    else:
        du_hash_str, frame_str, annotation_hash = values
    return uuid.UUID(du_hash_str), int(frame_str), annotation_hash


@router.get("/summary")
def route_project_summary(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    engine: Engine = Depends(dep_engine_readonly),
) -> metric_query.QuerySummary:
    tables = _get_metric_domain_tables(domain)
    with Session(engine) as sess:
        return metric_query.query_attr_summary(
            sess=sess,
            tables=tables,
            project_filters={
                "project_hash": [project_hash],
            },
            filters=filters,
        )


class AnalysisSearch(BaseModel):
    truncated: bool
    results: List[str]
    similarities: Optional[List[float]]


@functools.lru_cache
def get_embedder() -> CLIPEmbedder:
    return CLIPEmbedder()


@functools.lru_cache(maxsize=2)
def _get_similarity_query_index(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    engine: Engine,
) -> similarity_query.SimilarityQuery:
    tables = _get_metric_domain_tables(domain)
    base_domain: DomainTables = tables.primary
    with Session(engine) as sess:
        query = select(base_domain.metadata.embedding_clip, *base_domain.select_args(base_domain.metadata)).where(
            # FIXME: will break for nearest embedding on predictions
            base_domain.metadata.project_hash == project_hash,  # type: ignore
            is_not(base_domain.metadata.embedding_clip, None),
        )
        results = sess.exec(query).fetchall()
    embeddings = [np.frombuffer(e[0], dtype=np.float64) for e in results if e is not None]  # type: ignore
    return similarity_query.SimilarityQuery(embeddings, [rest for _em, *rest in results])  # type: ignore


def _find_similar_items(
    project_hash: uuid.UUID,
    src_embedding: bytes,
    domain: AnalysisDomain,
    tables: DomainTables,
    engine: Engine,
    similarity_exclude: Optional[DataOrAnnotateItem],
    extra_where: Optional[list],
    extra_where_has_filters: bool,
    limit: int,
) -> List[similarity_query.SimilarityResult]:
    similarity_query_impl = _get_similarity_query_index(project_hash, domain, engine)
    return similarity_query_impl.filter_query(
        engine=engine,
        tables=tables,
        project_hash=project_hash,
        filters=extra_where or [],
        has_filters=extra_where_has_filters,
        embedding=np.frombuffer(src_embedding, dtype=np.float64),
        k=limit,
        item=None if similarity_exclude is None else similarity_exclude,
    )


@router.post("/search")
def route_project_search(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    order_by: Optional[str] = Form(None),
    desc: bool = Form(),
    offset: int = Form(0, ge=0),
    limit: int = Form(1000, le=1000),
    engine: Engine = Depends(dep_engine_readonly),
    similarity_item: Optional[DataOrAnnotateItem] = Depends(parse_optional_data_or_annotate_item),
    text: Annotated[Optional[str], Form()] = None,
    image: Annotated[Optional[UploadFile], File()] = None,
) -> AnalysisSearch:
    tables = _get_metric_domain_tables(domain)
    base_table = tables.primary

    where = search_query.search_filters(
        tables=tables,
        base=base_table.analytics,
        search=filters,
        project_filters={"project_hash": [project_hash]},
    )
    where_has_filters = len(where) > 1  # 0 is always project hash constraint for 1 table

    similarity_searches = [
        True for similarity_search in [similarity_item, text, image] if similarity_search is not None
    ]

    is_similarity_query = False
    if len(similarity_searches) > 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Only one of `item`, `text` or `image` is allowed."
        )
    elif len(similarity_searches) == 1:
        is_similarity_query = True

    # Base implementation.
    if not is_similarity_query:
        with Session(engine) as sess:
            query = select(*[getattr(base_table.analytics, join_attr) for join_attr in base_table.join]).where(*where)

            if order_by is not None:
                if order_by in base_table.metrics or order_by in base_table.enums:
                    order_by_attr = getattr(base_table.analytics, order_by)
                    if desc:
                        order_by_attr = order_by_attr.desc()
                    query = query.order_by(order_by_attr)

            # + 1 to detect truncation.
            query = query.offset(offset).limit(limit + 1)
            search_results = sess.exec(query).fetchall()

        results = [
            _pack_id(**{str(join_attr): value for join_attr, value in zip(base_table.join, result)})
            for result in search_results[:limit]
        ]
        truncated = len(search_results) == limit + 1
        return AnalysisSearch(truncated=truncated, results=results, similarities=None)

    # Requested a similarity search implementation
    # Resolve to requested arguments to similarity enhanced search implementation
    similarity_embedding_bytes: Optional[bytes]
    similarity_exclude_constraint: Optional[DataOrAnnotateItem] = None
    if similarity_item is not None:
        join_attr_set = {
            "du_hash": similarity_item.du_hash,
            "frame": similarity_item.frame,
        }
        if similarity_item.annotation_hash is not None:
            join_attr_set["annotation_hash"] = similarity_item.annotation_hash
        elif domain == AnalysisDomain.Annotation:
            raise ValueError(f"Similarity domain mismatch: {similarity_item} in domain {domain}")
        with Session(engine) as sess:
            similarity_embedding_bytes = sess.exec(
                select(base_table.metadata.embedding_clip).where(
                    base_table.metadata.project_hash == project_hash,
                    *[
                        getattr(base_table.metadata, join_attr) == join_attr_set[join_attr]
                        for join_attr in base_table.join
                    ],
                    base_table.metadata.project_hash == project_hash,
                )
            ).first()
            similarity_exclude_constraint = similarity_item
    elif text is not None:
        similarity_embedding_bytes = get_embedder().embed_text(text).astype(dtype=np.float64).tobytes()
    elif image is not None:
        request_object_content = image.file.read()
        img_query = Image.open(io.BytesIO(request_object_content))
        img_exif = img_query.getexif()
        if img_exif and (274 in img_exif):  # 274 corresponds to orientation key for EXIF metadata
            img_query = ImageOps.exif_transpose(img_query)
        similarity_embedding_bytes = get_embedder().embed_image(img_query).astype(dtype=np.float64).tobytes()
    else:
        # NOTE: this shouldn't happen, but the type system doesn't understand this.
        raise ValueError("Missing search term")

    if similarity_embedding_bytes is None:
        raise RuntimeError("Similarity search against non missing item or failed to create embedding")

    similarity_result = _find_similar_items(
        project_hash,
        similarity_embedding_bytes,
        domain,
        base_table,
        engine,
        similarity_exclude_constraint,
        where,
        where_has_filters,
        limit + 1,
    )
    truncated = len(similarity_result) == limit + 1
    return AnalysisSearch(
        truncated=truncated,
        results=[r.item for r in similarity_result[:limit]],
        similarities=[r.similarity for r in similarity_result[:limit]],
    )


@router.get("/scatter")
def route_project_scatter(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    x_metric: str,
    y_metric: str,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(1000),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    engine: Engine = Depends(dep_engine_readonly),
) -> metric_query.QueryScatter:
    tables = _get_metric_domain_tables(domain)
    with Session(engine) as sess:
        return metric_query.query_attr_scatter(
            sess=sess,
            tables=tables,
            project_filters={"project_hash": [project_hash]},
            x_metric_name=x_metric,
            y_metric_name=y_metric,
            buckets=buckets,
            filters=filters,
        )


@router.get("/distribution")
def route_project_distribution(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    group: str,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(100),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    engine: Engine = Depends(dep_engine_readonly),
) -> metric_query.QueryDistribution:
    tables = _get_metric_domain_tables(domain)
    with Session(engine) as sess:
        return metric_query.query_attr_distribution(
            sess=sess,
            tables=tables,
            project_filters={
                "project_hash": [project_hash],
            },
            attr_name=group,
            buckets=buckets,
            filters=filters,
        )


@router.get("/reductions/{reduction_hash}/summary")
def route_project_reduction_scatter(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    reduction_hash: uuid.UUID,
    buckets: Literal[10, 100, 1000] = literal_bucket_depends(100),
    filters: search_query.SearchFiltersFastAPI = SearchFiltersFastAPIDepends,
    engine: Engine = Depends(dep_engine_readonly),
) -> metric_query.Query2DEmbedding:
    tables = _get_metric_domain_tables(domain)
    with Session(engine) as sess:
        return metric_query.query_reduction_scatter(
            sess=sess,
            tables=tables,
            project_filters={"project_hash": [project_hash], "reduction_hash": [reduction_hash]},
            buckets=buckets,
            filters=filters,
        )


# @router.get("/similarity/{item}")
# def route_project_similarity_search(
#     project_hash: uuid.UUID,
#     domain: AnalysisDomain,
#     embedding: Literal["embedding_clip"],
#     similarity_item: DataOrAnnotateItem = Depends(parse_data_or_annotate_item),
#     engine: Engine = Depends(dep_engine_readonly),
# ) -> List[similarity_query.SimilarityResult]:
#     tables = _get_metric_domain_tables(domain)
#     base_domain = tables.primary
#     if embedding != "embedding_clip":
#         raise ValueError("Unsupported embedding")
#     join_attr_set = {
#         "du_hash": similarity_item.du_hash,
#         "frame": similarity_item.frame,
#     }
#     if similarity_item.annotation_hash is not None:
#         join_attr_set["annotation_hash"] = similarity_item.annotation_hash
#     elif domain == AnalysisDomain.Annotation:
#         raise ValueError(f"Similarity domain mismatch: {similarity_item} in domain {domain}")
#
#     with Session(engine) as sess:
#         src_embedding = sess.exec(
#             select(base_domain.metadata.embedding_clip).where(
#                 base_domain.metadata.project_hash == project_hash,
#                 *[
#                     getattr(base_domain.metadata, join_attr) == join_attr_set[join_attr]
#                     for join_attr in base_domain.join
#                 ],
#             )
#         ).first()
#         if src_embedding is None:
#             raise ValueError("Source entry does not exist or missing embedding")
#         extra_where = (
#             functools.reduce(
#                 lambda a, b: a | b,
#                 [
#                     getattr(base_domain.metadata, join_attr) != join_attr_set[join_attr]
#                     for join_attr in base_domain.join
#                 ],
#             ),
#         )
#
#     return _find_similar_items(project_hash, src_embedding, domain, base_domain, engine, similarity_item, extra_where)


class MetricDissimilarityResult(BaseModel):
    results: Dict[str, float]


@router.get("/project_compare/metric_dissimilarity")
def route_project_compare_metric_dissimilarity(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    compare_project_hash: uuid.UUID,
    engine: Engine = Depends(dep_engine_readonly),
) -> MetricDissimilarityResult:
    tables = _get_metric_domain_tables(domain)
    base_domain = tables.primary
    dissimilarity = {}
    with Session(engine) as sess:
        for metric_name in base_domain.metrics.keys():
            metric_attr: float = getattr(base_domain.analytics, metric_name)
            all_data_1 = sess.exec(
                select(
                    metric_attr,
                ).where(base_domain.analytics.project_hash == project_hash, is_not(metric_attr, None))
            ).fetchall()
            all_data_2 = sess.exec(
                select(
                    metric_attr,
                ).where(base_domain.analytics.project_hash == compare_project_hash, is_not(metric_attr, None))
            ).fetchall()
            if len(all_data_1) > 0 and len(all_data_2) > 0:
                k_score, _ = ks_2samp(np.array(all_data_1), np.array(all_data_2))
                dissimilarity[metric_name] = k_score

    return MetricDissimilarityResult(
        results=dissimilarity,
    )
