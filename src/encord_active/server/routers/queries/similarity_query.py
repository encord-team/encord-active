import functools
import uuid
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from pydantic import BaseModel
from pynndescent import NNDescent
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from encord_active.db.models import (
    ProjectAnnotationAnalyticsExtra,
    ProjectDataAnalyticsExtra,
    ProjectPredictionAnalyticsExtra,
)
from encord_active.server.dependencies import (
    DataOrAnnotateItem,
    parse_data_or_annotate_item,
)
from encord_active.server.routers.queries.domain_query import DomainTables


class SimilarityResult(BaseModel):
    item: str
    similarity: float


def pack_similarity_result(
    k: Tuple[uuid.UUID, int, Optional[str]], s: Union[float, Dict[str, float]]
) -> SimilarityResult:
    du_hash, frame, *annotation_hash_opt = k
    annotation_hash = annotation_hash_opt[0] if len(annotation_hash_opt) > 0 else None
    item = DataOrAnnotateItem(du_hash=du_hash, frame=frame, annotation_hash=annotation_hash).pack()
    return SimilarityResult(
        item=item,
        similarity=s[item] if isinstance(s, dict) else s,
    )


def _sort_by_embedding(
    results: List[Union[ProjectDataAnalyticsExtra, ProjectAnnotationAnalyticsExtra, ProjectPredictionAnalyticsExtra]],
    embedding: np.ndarray,
    k: int,
) -> List[SimilarityResult]:
    embedding_stack = np.stack([np.frombuffer(r.embedding_clip or b"", dtype=np.float64) for r in results])
    embedding_distances = np.linalg.norm(embedding_stack - embedding, axis=1)
    embedding_indices: List[int] = np.argsort(embedding_distances)[:k].tolist()
    return [
        SimilarityResult(
            item=DataOrAnnotateItem(
                du_hash=results[i].du_hash,
                frame=results[i].frame,
                annotation_hash=getattr(results[i], "annotation_hash", None),
            ).pack(),
            similarity=embedding_distances[i],
        )
        for i in embedding_indices
    ]


def _filter_similarity_results(
    sess: Session,
    tables: DomainTables,
    embedding_order: List[SimilarityResult],
    filters: list,
    limit: int,
) -> List[SimilarityResult]:
    result = []
    for i in range(0, len(embedding_order), 200):
        embedding_slice = embedding_order[i : i + 200]
        embedding_slice_map = {s.item: s.similarity for s in embedding_slice}
        embedding_slice_unpacked = [parse_data_or_annotate_item(s.item) for s in embedding_slice]
        # Fetch all that match WITH filters
        filtered_res: "List[Union[tuple[uuid.UUID, int], tuple[uuid.UUID, int, str]]]" = sess.exec(
            select(*tables.select_args(tables.analytics)).where(
                *filters,
                functools.reduce(
                    lambda a, b: a | b,
                    [
                        functools.reduce(
                            lambda av, bv: av & bv,
                            [getattr(tables.analytics, j) == getattr(embedding_tuple, j) for j in tables.join],
                        )
                        for embedding_tuple in embedding_slice_unpacked
                    ],
                ),
            )
        ).fetchall()
        if len(filtered_res) > len(embedding_slice_unpacked):
            raise RuntimeError(
                f"Bug with filtered search implementation: {len(filtered_res)} > {len(embedding_slice_unpacked)}"
            )
        filtered_slice: List[SimilarityResult] = [
            pack_similarity_result(tuple(r), embedding_slice_map) for r in filtered_res  # type: ignore
        ]
        filtered_slice.sort(key=lambda v: v.similarity)
        result += filtered_slice
        if len(result) >= limit:
            break
    return result[:limit]


class SimilarityQuery:
    query_impl: Union[np.ndarray, NNDescent]
    results: List[Tuple[uuid.UUID, int, Optional[str]]]

    def __init__(self, embeddings: List[np.ndarray], results: List[Tuple[uuid.UUID, int, Optional[str]]]) -> None:
        embeddings_stack: np.ndarray = np.stack(embeddings)
        if len(embeddings) <= 4096:
            self.query_impl = embeddings_stack
        else:
            nn: NNDescent = NNDescent(data=embeddings_stack, metric="euclidean")  # Euclidean distance
            nn.prepare()
            self.query_impl = nn
        self.results = results

    def query(self, embedding: np.ndarray, k: int, item: Optional[DataOrAnnotateItem]) -> List[SimilarityResult]:
        if isinstance(self.query_impl, NNDescent):
            indices_stack, distances_stack = self.query_impl.query(embedding.reshape(1, -1), k=k)
            indices = indices_stack.reshape(-1)
            distances = distances_stack.reshape(-1)
        else:
            offsets = self.query_impl - embedding
            distances = np.linalg.norm(offsets, axis=1)  # Euclidean distance
            indices = np.argsort(distances)[:k]
            distances = distances[indices]
        similarity_results = [
            pack_similarity_result(self.results[idx], similarity) for idx, similarity in zip(indices, distances)
        ]
        if item and len(similarity_results) > 0:
            similarity_0 = similarity_results[0].item
            if similarity_0 == item.pack():
                similarity_results = similarity_results[1:]

        return similarity_results

    def filter_query(
        self,
        engine: Engine,
        tables: DomainTables,
        filters: list,
        has_filters: bool,
        project_hash: uuid.UUID,
        embedding: np.ndarray,
        k: int,
        item: Optional[DataOrAnnotateItem],
    ) -> List[SimilarityResult]:
        # Fallback implementation for other database engines.
        # 3 possible strategies:
        #   1. Pure similarity search. (No filters ONLY).
        #   2. Apply filters, fetch ALL, then sort by calculated similarities
        #   3. Generate similarity search order (2x limit for 50% filter ratio) & apply filters on results.
        if not has_filters:
            # Strategy 1:
            return self.query(embedding, k, item)
        with Session(engine) as sess:
            # Try strategy 2, with limit of max('k', 10_000). If not truncated, use this strategy.
            s2_limit = max(k, 10_000)
            s2_where = (
                filters
                + [getattr(tables.analytics, j) == getattr(tables.metadata, j) for j in tables.join]
                + [tables.metadata.project_hash == project_hash]
            )
            s2_results: List[
                Union[ProjectDataAnalyticsExtra, ProjectAnnotationAnalyticsExtra, ProjectPredictionAnalyticsExtra]
            ] = sess.exec(
                select(tables.metadata).where(*s2_where).limit(s2_limit)
            ).fetchall()  # type: ignore
            if len(s2_results) < s2_limit:
                return _sort_by_embedding(s2_results, embedding, k)

            # Try strategy 3, with limit of 2 * k - so that this strategy works with a 50% filtering rate.
            s3_mul: int = 2
            s3_search_order = self.query(embedding, s3_mul * k, item)
            s3_filtered = _filter_similarity_results(sess, tables, s3_search_order, filters, k)
            if len(s3_filtered) >= k or len(s3_search_order) < s3_mul * k:
                return s3_filtered[:k]

            # Fallback - use strategy 2 with NO limit on number of results returned.
            # Will always return the correct result, but increases memory usage.
            fallback_results: List[
                Union[ProjectDataAnalyticsExtra, ProjectAnnotationAnalyticsExtra, ProjectPredictionAnalyticsExtra]
            ] = sess.exec(
                select(tables.metadata).where(*s2_where).limit(s2_limit)
            ).fetchall()  # type: ignore
            return _sort_by_embedding(fallback_results, embedding, k)


def pg_similarity_query(
    sess: Session,
    tables: DomainTables,
    project_hash: uuid.UUID,
    embedding: bytes,
    extra_where: Optional[list],
    exclude_item: Optional[DataOrAnnotateItem],
    limit: int,
) -> List[SimilarityResult]:
    pg_where = list(extra_where or [])
    if len(pg_where) > 0:
        pg_where = pg_where + [getattr(tables.analytics, j) == getattr(tables.metadata, j) for j in tables.join]
    if exclude_item is not None:
        pg_where.append(
            functools.reduce(
                lambda a, b: a | b, [getattr(tables.metadata, j) != getattr(exclude_item, j) for j in tables.join]
            )
        )
    pg_query = (
        select(
            tables.metadata.embedding_clip.l2_distance(embedding).label("similarity"),  # type: ignore
            *tables.select_args(tables.metadata),
        )
        .where(
            tables.metadata.project_hash == project_hash,
            *pg_where,
        )
        .order_by("similarity")
        .limit(limit)
    )
    pg_results = sess.exec(pg_query).fetchall()
    return [
        pack_similarity_result(tuple(similarity_item), similarity)  # type: ignore
        for similarity, *similarity_item in pg_results
    ]
