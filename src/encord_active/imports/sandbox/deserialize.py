import base64
import json
from pathlib import Path
from typing import Optional

from sqlalchemy.engine import Engine
from sqlmodel import Session

from encord_active.exports.serialize import WholeProjectModelV1


def import_serialized_project(engine: Engine, db_json: Path) -> None:
    raw_json = json.loads(db_json.read_bytes())

    def decode_bytes(encoded: Optional[bytes]) -> Optional[bytes]:
        if encoded is None:
            return None
        return base64.b85decode(encoded)

    for elem in raw_json.get("project_analytics_data_extra", []):
        if "embedding_clip" in elem:
            elem["embedding_clip"] = decode_bytes(elem["embedding_clip"])
    for elem in raw_json.get("project_analytics_annotation_extra", []):
        if "embedding_clip" in elem:
            elem["embedding_clip"] = decode_bytes(elem["embedding_clip"])
        if "embedding_hu" in elem:
            elem["embedding_hu"] = decode_bytes(elem["embedding_hu"])
    for elem in raw_json.get("prediction_analytics_extra", []):
        if "embedding_clip" in elem:
            elem["embedding_clip"] = decode_bytes(elem["embedding_clip"])
        if "embedding_hu" in elem:
            elem["embedding_hu"] = decode_bytes(elem["embedding_hu"])
    for elem in raw_json.get("project_embedding_reduction", []):
        if "reduction_bytes" in elem:
            elem["reduction_bytes"] = decode_bytes(elem["reduction_bytes"])
    whole_project_json = WholeProjectModelV1.parse_obj(raw_json)

    with Session(engine) as sess:
        sess.bulk_save_objects(whole_project_json.project)
        sess.bulk_save_objects(whole_project_json.project_import)
        sess.bulk_save_objects(whole_project_json.project_data)
        sess.bulk_save_objects(whole_project_json.project_data_units)
        sess.bulk_save_objects(whole_project_json.project_collaborator)
        sess.bulk_save_objects(whole_project_json.project_embedding_reduction)
        sess.bulk_save_objects(whole_project_json.project_embedding_index)
        sess.bulk_save_objects(whole_project_json.project_analytics_data)
        sess.bulk_save_objects(whole_project_json.project_analytics_data_extra)
        sess.bulk_save_objects(whole_project_json.project_analytics_data_derived)
        sess.bulk_save_objects(whole_project_json.project_analytics_data_reduced)
        sess.bulk_save_objects(whole_project_json.project_analytics_annotation)
        sess.bulk_save_objects(whole_project_json.project_analytics_annotation_extra)
        sess.bulk_save_objects(whole_project_json.project_analytics_annotation_derived)
        sess.bulk_save_objects(whole_project_json.project_analytics_annotation_reduced)
        sess.bulk_save_objects(whole_project_json.project_tags)
        sess.bulk_save_objects(whole_project_json.project_tagged_data)
        sess.bulk_save_objects(whole_project_json.project_tagged_annotation)
        sess.bulk_save_objects(whole_project_json.prediction)
        sess.bulk_save_objects(whole_project_json.prediction_data)
        sess.bulk_save_objects(whole_project_json.prediction_data_units)
        sess.bulk_save_objects(whole_project_json.prediction_analytics)
        sess.bulk_save_objects(whole_project_json.prediction_analytics_extra)
        sess.bulk_save_objects(whole_project_json.prediction_analytics_derived)
        sess.bulk_save_objects(whole_project_json.prediction_analytics_reduced)
        sess.bulk_save_objects(whole_project_json.prediction_analytics_fn)
        sess.bulk_save_objects(whole_project_json.project_tagged_prediction)
        sess.commit()
