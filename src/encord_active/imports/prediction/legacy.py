import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Union

from encord.objects import Object, OntologyStructure
from pydantic import parse_file_as

from encord_active.db.enums import AnnotationType
from encord_active.db.models import ProjectPrediction
from encord_active.db.models.project import Project
from encord_active.imports.prediction.op import (
    PredictionImportSpec,
    ProjectPredictedDataUnit,
)
from encord_active.imports.util import (
    append_object_to_list,
    bitmask_to_bounding_box,
    bitmask_to_encord_dict,
    bitmask_to_polygon,
    bitmask_to_rotatable_bounding_box,
    coco_str_to_bitmask,
)
from encord_active.lib.db.predictions import BoundingBox, Format, Prediction


def import_legacy_predictions(
    ontology: dict,
    prediction_name: str,
    project_hash: uuid.UUID,
    prediction_file: Path,
) -> PredictionImportSpec:
    predictions: List[Prediction] = parse_file_as(List[Prediction], prediction_file, allow_pickle=True)

    ontology_structure = OntologyStructure.from_dict(ontology)

    # Process the file
    du_prediction_list: Dict[Tuple[uuid.UUID, int], ProjectPredictedDataUnit] = {}
    for predict in predictions:
        du_hash = uuid.UUID(predict.data_hash)
        frame = predict.frame or 0
        predicted_data_unit = du_prediction_list.setdefault(
            (du_hash, frame),
            ProjectPredictedDataUnit(
                du_hash=du_hash,
                frame=frame,
                objects=[],
                classifications=[],
            ),
        )

        if predict.object:
            shape = predict.object.data
            if predict.object.format == Format.POLYGON:
                annotation_type = AnnotationType.POLYGON
                raise TypeError("Polygon predictions currently not supported")
            elif predict.object.format == Format.BOUNDING_BOX:
                annotation_type = AnnotationType.BOUNDING_BOX
                assert isinstance(predict.object.data, BoundingBox)
                shape = predict.object.data.dict()
                ontology_object = ontology_structure.get_child_by_hash(predict.object.feature_hash)
            else:
                annotation_type = AnnotationType.BITMASK
                raise TypeError("Bitmas predictions currently not supported")
        elif predict.classification:
            raise TypeError("Classification predictions currently not supported")
        else:
            raise TypeError("Mismatched prediction types. All predictions must be either objects or classifications")

        append_object_to_list(
            object_list=predicted_data_unit.objects,
            annotation_type=annotation_type,
            shape_data_list=[shape],
            ontology_object=ontology_object,
            confidence=predict.confidence,
            object_hash=None,
        )

    prediction_hash = uuid.uuid4()
    return PredictionImportSpec(
        prediction=ProjectPrediction(
            prediction_hash=prediction_hash,
            name=prediction_name,
            project_hash=project_hash,
            external_project_hash=None,
        ),
        du_prediction_list=list(du_prediction_list.values()),
    )
