import uuid
from typing import Dict, List

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from encord_active.db.models import (
    ProjectDataUnitMetadata,
    ProjectTag,
    ProjectTaggedAnnotation,
    ProjectTaggedDataUnit,
)
from encord_active.lib.db.helpers.tags import GroupedTags
from encord_active.server.dependencies import dep_engine

router = APIRouter(
    prefix="/{project_hash}/tags",
)


@router.get("/tagged_items")
def route_tagged_items(project_hash: uuid.UUID, engine: Engine = Depends(dep_engine)) -> Dict[str, GroupedTags]:
    identifier_tags: dict[str, GroupedTags] = {}
    with Session(engine) as sess:
        data_tags = sess.exec(
            select(ProjectTaggedDataUnit.du_hash, ProjectTaggedDataUnit.frame, ProjectTag.name,).where(
                ProjectTaggedDataUnit.tag_hash == ProjectTag.tag_hash,
                ProjectTaggedDataUnit.project_hash == project_hash,
                ProjectTag.project_hash == project_hash,
                ProjectDataUnitMetadata.project_hash == project_hash,
                ProjectDataUnitMetadata.du_hash == ProjectTaggedDataUnit.du_hash,
                ProjectDataUnitMetadata.frame == ProjectTaggedDataUnit.frame,
            )
        ).all()
        for du_hash, frame, tag in data_tags:
            key = f"{du_hash}_{frame}"
            identifier_tags.setdefault(key, GroupedTags(data=[], label=[]))["data"].append(tag)
        label_tags = sess.exec(
            select(
                ProjectTaggedAnnotation.du_hash,
                ProjectTaggedAnnotation.frame,
                ProjectTaggedAnnotation.annotation_hash,
                ProjectTag.name,
            ).where(
                ProjectTaggedAnnotation.tag_hash == ProjectTag.tag_hash,
                ProjectTaggedAnnotation.project_hash == project_hash,
                ProjectTag.project_hash == project_hash,
                ProjectDataUnitMetadata.project_hash == project_hash,
                ProjectDataUnitMetadata.du_hash == ProjectTaggedAnnotation.du_hash,
                ProjectDataUnitMetadata.frame == ProjectTaggedAnnotation.frame,
            )
        ).all()
        for du_hash, frame, annotation_hash, tag in label_tags:
            data_key = f"{du_hash}_{frame:05d}"
            label_key = f"{data_key}_{annotation_hash}"
            du_tags = identifier_tags.setdefault(data_key, GroupedTags(data=[], label=[]))
            if tag not in du_tags["label"]:
                du_tags["label"].append(tag)
            identifier_tags.setdefault(label_key, GroupedTags(data=du_tags["data"], label=[]))["label"].append(tag)
    return identifier_tags


class ItemTags(BaseModel):
    id: str
    grouped_tags: GroupedTags


@router.put("/tag_items")
def route_tag_items(project_hash: uuid.UUID, payload: List[ItemTags], engine: Engine = Depends(dep_engine)):
    with Session(engine) as sess:

        def _get_or_create_tag_hash(name: str) -> uuid.UUID:
            tag_hash_candidate = sess.exec(
                select(ProjectTag.tag_hash).where(ProjectTag.project_hash == project_hash, ProjectTag.name == name)
            ).first()
            if tag_hash_candidate is not None:
                return tag_hash_candidate
            else:
                new_tag_hash = uuid.uuid4()
                sess.add(
                    ProjectTag(
                        tag_hash=new_tag_hash,
                        project_hash=project_hash,
                        name=name,
                        description="",
                    )
                )
                return new_tag_hash

        data_exists = set()
        label_exists = set()
        for item in payload:
            data_tag_list = item.grouped_tags["data"]
            annotation_tag_list = item.grouped_tags["label"]
            du_hash_str, frame_str, *annotation_hashes = item.id.split("_")
            du_hash = uuid.UUID(du_hash_str)
            frame = int(frame_str)
            existing_data_tags = set(
                sess.exec(
                    select(ProjectTaggedDataUnit.tag_hash).where(
                        ProjectTaggedDataUnit.project_hash == project_hash,
                        ProjectTaggedDataUnit.du_hash == du_hash,
                        ProjectTaggedDataUnit.frame == frame,
                    )
                ).all()
            )
            new_data_tag_uuids: set[uuid.UUID] = set()
            for data_tag in data_tag_list:
                tag_hash = _get_or_create_tag_hash(data_tag)
                dup_key = (project_hash, du_hash, frame, tag_hash)
                if dup_key in data_exists:
                    continue
                data_exists.add(dup_key)
                new_data_tag_uuids.add(tag_hash)
                if tag_hash in existing_data_tags:
                    continue
                sess.add(
                    ProjectTaggedDataUnit(
                        project_hash=project_hash,
                        du_hash=du_hash,
                        frame=frame,
                        tag_hash=tag_hash,
                    )
                )
            for tag_hash in existing_data_tags.difference(new_data_tag_uuids):
                data_tag_to_delete = sess.exec(
                    select(ProjectTaggedDataUnit).where(
                        ProjectTaggedDataUnit.project_hash == project_hash,
                        ProjectTaggedDataUnit.du_hash == du_hash,
                        ProjectTaggedDataUnit.frame == frame,
                        ProjectTaggedDataUnit.tag_hash == tag_hash,
                    )
                ).first()
                sess.delete(data_tag_to_delete)

            for annotation_hash in annotation_hashes:
                existing_label_tags = set(
                    sess.exec(
                        select(ProjectTaggedAnnotation.tag_hash).where(
                            ProjectTaggedAnnotation.project_hash == project_hash,
                            ProjectTaggedAnnotation.du_hash == du_hash,
                            ProjectTaggedAnnotation.frame == frame,
                            ProjectTaggedAnnotation.annotation_hash == annotation_hash,
                        )
                    )
                )
                new_label_tag_uuids: set[uuid.UUID] = set()
                for annotation_tag in annotation_tag_list:
                    tag_hash = _get_or_create_tag_hash(annotation_tag)
                    dup_key2 = (project_hash, du_hash, frame, annotation_hash, tag_hash)
                    if dup_key2 in label_exists:
                        continue
                    new_label_tag_uuids.add(tag_hash)
                    label_exists.add(dup_key2)
                    if tag_hash in existing_label_tags:
                        continue
                    sess.add(
                        ProjectTaggedAnnotation(
                            project_hash=project_hash,
                            du_hash=du_hash,
                            frame=frame,
                            annotation_hash=annotation_hash,
                            tag_hash=tag_hash,
                        )
                    )
                for tag_hash in existing_label_tags.difference(new_label_tag_uuids):
                    tag_to_remove = sess.exec(
                        select(ProjectTaggedAnnotation).where(
                            ProjectTaggedAnnotation.project_hash == project_hash,
                            ProjectTaggedAnnotation.du_hash == du_hash,
                            ProjectTaggedAnnotation.frame == frame,
                            ProjectTaggedAnnotation.annotation_hash == annotation_hash,
                            ProjectTaggedAnnotation.tag_hash == tag_hash,
                        )
                    ).first()
                    sess.delete(tag_to_remove)
        sess.commit()
