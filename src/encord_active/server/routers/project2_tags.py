import uuid
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import Text, delete, insert, literal, tuple_, func
from sqlalchemy.engine import Dialect, Engine
from sqlalchemy.sql.operators import in_op
from sqlmodel import Session, select
from sqlmodel.sql.sqltypes import GUID

from encord_active.db.models import (
    ProjectTag,
    ProjectTaggedAnnotation,
    ProjectTaggedDataUnit,
)
from encord_active.db.util.char8 import Char8
from encord_active.server.dependencies import (
    DataOrAnnotateItem,
    dep_engine,
    dep_engine_readonly,
    parse_data_or_annotate_item,
)
from encord_active.server.routers.queries import search_query
from encord_active.server.routers.queries.domain_query import (
    TABLES_ANNOTATION,
    TABLES_DATA,
)
from encord_active.server.routers.route_tags import AnalysisDomain, RouteTag

router = APIRouter(
    prefix="/{project_hash}/tags",
    tags=[RouteTag.PROJECT, RouteTag.PREDICTION],
)


class GroupedTags(BaseModel):
    data: List[str]
    label: List[str]


class ProjectTagEntry(BaseModel):
    hash: uuid.UUID
    name: str


class ProjectTagEntryMeta(BaseModel):
    hash: uuid.UUID
    name: str
    description: str
    dataCount: int
    labelCount: int
    createdAt: datetime
    lastEditedAt: datetime


@router.get("/")
def route_list_tags(project_hash: uuid.UUID, engine: Engine = Depends(dep_engine_readonly)) -> List[ProjectTagEntry]:
    with Session(engine) as sess:
        tags = sess.exec(
            select(ProjectTag.tag_hash, ProjectTag.name).where(ProjectTag.project_hash == project_hash)
        ).fetchall()
    return [ProjectTagEntry(hash=tag_hash, name=name) for tag_hash, name in tags]

@router.get("/meta")
def route_list_tags(project_hash: uuid.UUID, engine: Engine = Depends(dep_engine_readonly)) -> List[ProjectTagEntryMeta]:
    with Session(engine) as sess:
        tags = sess.exec(
            select(ProjectTag.tag_hash, ProjectTag.name, ProjectTag.description, func.count(ProjectTaggedDataUnit.du_hash), ProjectTag.created_at, ProjectTag.last_edited_at ).where(ProjectTag.project_hash == project_hash)
            .outerjoin( ProjectTaggedDataUnit, ProjectTag.tag_hash == ProjectTaggedDataUnit.tag_hash)
            .group_by(ProjectTag.tag_hash )
        ).fetchall()
    return[ProjectTagEntryMeta(hash=tag_hash, name=name, description=description, dataCount=dataCount, labelCount=32, createdAt = created_at, lastEditedAt=last_edited_at) for tag_hash, name, description, dataCount, created_at, last_edited_at in tags]


class ProjectTagRequest(BaseModel):
    name: str
    description: str

@router.post("/")
def route_create_tags(
    project_hash: uuid.UUID, tags: List[ProjectTagRequest], engine=Depends(dep_engine)
) -> Dict[str, uuid.UUID]:
    with Session(engine) as sess:
        project_tags = [
            ProjectTag(tag_hash=uuid.uuid4(), name=tag.name, description=tag.description, project_hash=project_hash, created_at=datetime.now(), last_edited_at=datetime.now())
            for tag in tags
        ]
        sess.add_all(project_tags)
        sess.commit()
        return {tag.name: tag.tag_hash for tag in project_tags}


def _where_tag_items(dialect: Dialect, items: List[str]) -> Tuple[Optional[list], Optional[list]]:
    unpacked_items: List[DataOrAnnotateItem] = [parse_data_or_annotate_item(item) for item in items]
    unpacked_data: List[Tuple[uuid.UUID, int]] = [
        (unpacked_item.du_hash, unpacked_item.frame)
        for unpacked_item in unpacked_items
        if unpacked_item.annotation_hash is None
    ]
    unpacked_annotation: List[Tuple[uuid.UUID, int, str]] = [
        (unpacked_item.du_hash, unpacked_item.frame, unpacked_item.annotation_hash)
        for unpacked_item in unpacked_items
        if unpacked_item.annotation_hash is not None
    ]
    if dialect.name == "postgresql":
        pg_data_list = None
        pg_annotation_list = None
        if len(unpacked_data) > 0:
            pg_data_list = [in_op(tuple_(ProjectTaggedDataUnit.du_hash, ProjectTaggedDataUnit.frame), unpacked_data)]
        if len(unpacked_annotation) > 0:
            pg_annotation_list = [
                in_op(
                    tuple_(
                        ProjectTaggedAnnotation.du_hash,
                        ProjectTaggedAnnotation.frame,
                        ProjectTaggedAnnotation.annotation_hash,
                    ),
                    unpacked_annotation,
                )
            ]
        return pg_data_list, pg_annotation_list

    # Sqlite fallback implementation
    sqlite_data_list = None
    sqlite_annotation_list = None
    guid = GUID().bind_processor(dialect)
    if len(unpacked_data) > 0:
        sqlite_data_list = [
            in_op(ProjectTaggedDataUnit.du_hash, list({du_hash for du_hash, _ in unpacked_data})),
            in_op(ProjectTaggedDataUnit.frame, list({frame for _, frame in unpacked_data})),
            in_op(
                ProjectTaggedDataUnit.du_hash.cast(Text).concat("_").concat(ProjectTaggedDataUnit.frame.cast(Text)),  # type: ignore
                [f"{guid(du_hash)}_{frame}" for du_hash, frame in unpacked_data],
            ),
        ]
    if len(unpacked_annotation) > 0:
        char8 = Char8().bind_processor(dialect)
        sqlite_annotation_list = [
            in_op(ProjectTaggedAnnotation.du_hash, list({du_hash for du_hash, _, _ in unpacked_annotation})),
            in_op(ProjectTaggedAnnotation.frame, list({frame for _, frame, _ in unpacked_annotation})),
            in_op(ProjectTaggedAnnotation.annotation_hash, list({ah for _, _, ah in unpacked_annotation})),
            in_op(
                ProjectTaggedAnnotation.du_hash.cast(Text).concat("_").concat(ProjectTaggedAnnotation.frame.cast(Text)).concat("_").concat(ProjectTaggedAnnotation.annotation_hash.cast(Text)),  # type: ignore
                [f"{guid(du_hash)}_{frame}_{char8(ah)}" for du_hash, frame, ah in unpacked_annotation],
            ),
        ]

    return sqlite_data_list, sqlite_annotation_list


class AllTagsResult(BaseModel):
    data: List[ProjectTagEntry]
    annotation: List[ProjectTagEntry]


@router.post("/items/all_tags")
def route_items_all_tags(
    project_hash: uuid.UUID, items: List[str], engine: Engine = Depends(dep_engine_readonly)
) -> AllTagsResult:
    where_tag_data, where_tag_annotation = _where_tag_items(engine.dialect, items)
    with Session(engine) as sess:
        data_tags = None
        if where_tag_data is not None:
            data_tags = sess.exec(
                select(ProjectTag.tag_hash, ProjectTag.name)
                .where(
                    ProjectTag.project_hash == project_hash,
                    ProjectTaggedDataUnit.project_hash == project_hash,
                    ProjectTag.tag_hash == ProjectTaggedDataUnit.tag_hash,
                    *where_tag_data,
                )
                .group_by(ProjectTag.tag_hash, ProjectTag.name)
            ).fetchall()
        annotation_tags = None
        if where_tag_annotation is not None:
            annotation_tags = sess.exec(
                select(ProjectTag.tag_hash, ProjectTag.name)
                .where(
                    ProjectTag.project_hash == project_hash,
                    ProjectTaggedAnnotation.project_hash == project_hash,
                    ProjectTag.tag_hash == ProjectTaggedAnnotation.tag_hash,
                    *where_tag_annotation,
                )
                .group_by(ProjectTag.tag_hash, ProjectTag.name)
            ).fetchall()
    return AllTagsResult(
        data=[ProjectTagEntry(hash=tag_hash, name=name) for tag_hash, name in data_tags] if data_tags else [],
        annotation=[ProjectTagEntry(hash=tag_hash, name=name) for tag_hash, name in annotation_tags]
        if annotation_tags
        else [],
    )


@router.post("/items/tag_all")
def route_items_tag_all(
    project_hash: uuid.UUID,
    items: List[str],
    tags: List[uuid.UUID],
    engine: Engine = Depends(dep_engine),
) -> None:
    # FIXME: assert domain equality?!
    parsed_items = [parse_data_or_annotate_item(item) for item in items]
    with Session(engine) as sess:
        for tag in tags:
            for item in parsed_items:
                if item.annotation_hash is not None:
                    sess.merge(
                        ProjectTaggedAnnotation(
                            project_hash=project_hash,
                            du_hash=item.du_hash,
                            frame=item.frame,
                            annotation_hash=item.annotation_hash,
                            tag_hash=tag,
                        )
                    )
                else:
                    sess.merge(
                        ProjectTaggedDataUnit(
                            project_hash=project_hash,
                            du_hash=item.du_hash,
                            frame=item.frame,
                            tag_hash=tag,
                        )
                    )
        sess.commit()


@router.post("/items/untag_all")
def route_items_untag_all(
    project_hash: uuid.UUID,
    items: List[str],
    tags: List[uuid.UUID],
    engine: Engine = Depends(dep_engine),
) -> None:
    where_tag_data, where_tag_annotation = _where_tag_items(engine.dialect, items)
    with Session(engine) as sess:
        if engine.dialect.name == "postgresql":
            if where_tag_data is not None:
                sess.execute(
                    delete(ProjectTaggedDataUnit).where(
                        ProjectTaggedDataUnit.project_hash == project_hash,
                        in_op(ProjectTaggedDataUnit.tag_hash, tags),
                        *where_tag_data,
                    )
                )
            if where_tag_annotation is not None:
                sess.execute(
                    delete(ProjectTaggedAnnotation).where(
                        ProjectTaggedAnnotation.project_hash == project_hash,
                        in_op(ProjectTaggedAnnotation.tag_hash, tags),
                        *where_tag_annotation,
                    )
                )
        else:
            if where_tag_data is not None:
                data_tags = sess.exec(
                    select(ProjectTaggedDataUnit).where(
                        ProjectTaggedDataUnit.project_hash == project_hash,
                        in_op(ProjectTaggedDataUnit.tag_hash, tags),
                        *where_tag_data,
                    )
                ).fetchall()
                for data_tag in data_tags:
                    sess.delete(data_tag)
            if where_tag_annotation is not None:
                annotation_tags = sess.exec(
                    select(ProjectTaggedAnnotation).where(
                        ProjectTaggedAnnotation.project_hash == project_hash,
                        in_op(ProjectTaggedAnnotation.tag_hash, tags),
                        *where_tag_annotation,
                    )
                ).fetchall()
                for annotation_tag in annotation_tags:
                    sess.delete(annotation_tag)
        sess.commit()


@router.post("/{domain}/filter/all_tags")
def route_filter_all_tags(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    filters: search_query.SearchFiltersFastAPI,
    engine: Engine = Depends(dep_engine_readonly),
) -> AllTagsResult:
    where_data = search_query.search_filters(
        tables=TABLES_DATA,
        base=TABLES_DATA.primary.tag,
        search=filters,
        project_filters={"project_hash": [project_hash]},
    )
    where_annotation = search_query.search_filters(
        tables=TABLES_ANNOTATION,
        base=TABLES_ANNOTATION.primary.tag,
        search=filters,
        project_filters={"project_hash": [project_hash]},
    )
    with Session(engine) as sess:
        data_tags = sess.exec(
            select(ProjectTag.tag_hash, ProjectTag.name)
            .where(*where_data, ProjectTag.tag_hash == ProjectTaggedDataUnit.tag_hash)
            .group_by(ProjectTag.tag_hash, ProjectTag.name)
        ).fetchall()
        if domain == AnalysisDomain.Data:
            # FIXME: potentially re-enable this (disabled for consistency with item variant)
            annotation_tags = []
        else:
            annotation_tags = sess.exec(
                select(ProjectTag.tag_hash, ProjectTag.name)
                .where(*where_annotation, ProjectTag.tag_hash == ProjectTaggedAnnotation.tag_hash)
                .group_by(ProjectTag.tag_hash, ProjectTag.name)
            ).fetchall()

    return AllTagsResult(
        data=[ProjectTagEntry(hash=tag_hash, name=name) for tag_hash, name in data_tags],
        annotation=[ProjectTagEntry(hash=tag_hash, name=name) for tag_hash, name in annotation_tags],
    )


@router.post("/{domain}/filter/tag_all")
def route_filter_tag_all(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    tags: List[uuid.UUID],
    filters: search_query.SearchFiltersFastAPI,
    engine: Engine = Depends(dep_engine),
) -> None:
    tables = TABLES_ANNOTATION if domain == AnalysisDomain.Annotation else TABLES_DATA
    where = search_query.search_filters(
        tables=tables,
        base=tables.primary.analytics,  # NOTE: technically should use ANY
        search=filters,
        project_filters={"project_hash": [project_hash]},
    )
    with Session(engine) as sess:
        # FIXME: make this 1 query!!!
        guid = GUID().bind_processor(engine.dialect)
        for tag_hash in tags:
            sess.execute(
                insert(tables.primary.tag).from_select(
                    ["project_hash", "tag_hash"] + tables.primary.join,
                    select(  # type: ignore
                        literal(guid(project_hash)),
                        literal(guid(tag_hash)),
                        *[getattr(tables.primary.analytics, j) for j in tables.primary.join],
                    ).where(*where),
                )
            )
        sess.commit()


@router.post("/{domain}/filter/untag_all")
def route_filter_untag_all(
    project_hash: uuid.UUID,
    domain: AnalysisDomain,
    tags: List[uuid.UUID],
    filters: search_query.SearchFiltersFastAPI,
    engine: Engine = Depends(dep_engine),
) -> None:
    tables = TABLES_ANNOTATION if domain == AnalysisDomain.Annotation else TABLES_DATA
    where = search_query.search_filters(
        tables=tables, base=tables.primary.tag, search=filters, project_filters={"project_hash": [project_hash]}
    )
    with Session(engine) as sess:
        sess.execute(
            delete(tables.primary.tag).where(
                in_op(tables.primary.tag.tag_hash, tags),
                *where,
            )
        )
        sess.commit()
