import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from encord.objects import Object, OntologyStructure
from pydantic import BaseModel, parse_file_as

from encord_active.db.enums import AnnotationType
from encord_active.db.models import ProjectPrediction
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


class CoCoPredictionSegmentationBitmask(BaseModel):
    size: Tuple[int, int]
    counts: str


class CoCoPrediction(BaseModel):
    image_id: int
    category_id: int
    segmentation: CoCoPredictionSegmentationBitmask
    score: float
    # Encord:
    encord_track_uuid: Optional[str] = None


def import_coco_result(
    ontology: dict,
    import_metadata: dict,
    prediction_name: str,
    project_hash: uuid.UUID,
    prediction_file: Path,
) -> PredictionImportSpec:
    predictions: List[CoCoPrediction] = parse_file_as(List[CoCoPrediction], prediction_file)
    # Generate reverse category lookup dictionary.
    ontology_structure = OntologyStructure.from_dict(ontology)
    feature_hash_map: Dict[int, Tuple[AnnotationType, Object]] = {}
    for obj in ontology_structure.objects:
        feature_hash_map[obj.uid] = AnnotationType(obj.shape.value), obj

    # Generate image reverse lookup dictionary.

    # Process the file
    import_metadata_images = import_metadata["images"]
    du_prediction_list: Dict[Tuple[uuid.UUID, int], ProjectPredictedDataUnit] = {}
    for predict in predictions:
        du_hash = uuid.UUID(import_metadata_images[str(predict.image_id)])
        if (du_hash, 0) not in du_prediction_list:
            du_prediction_list[du_hash, 0] = ProjectPredictedDataUnit(
                du_hash=du_hash,
                frame=0,
                objects=[],
                classifications=[],
            )
        predicted_data_unit = du_prediction_list[du_hash, 0]
        if (predict.category_id + 1) not in feature_hash_map:
            raise ValueError(
                f"WARNING: category_id={predict.category_id} (+1) is not in the project ontology, skipping prediction"
            )
        annotation_type, ontology_object = feature_hash_map[predict.category_id + 1]
        bitmask = coco_str_to_bitmask(
            predict.segmentation.counts, width=predict.segmentation.size[0], height=predict.segmentation.size[1]
        )
        # Counts to torch tensor
        shape_dict_list: List[Union[Dict[str, float], Dict[str, Dict[str, float]], str]]
        if annotation_type == AnnotationType.BOUNDING_BOX:
            shape_dict_list = [bitmask_to_bounding_box(bitmask)]
        elif annotation_type == AnnotationType.ROTATABLE_BOUNDING_BOX:
            shape_dict_list = [bitmask_to_rotatable_bounding_box(bitmask)]
        elif annotation_type == AnnotationType.POLYGON:
            shape_dict_list = list(bitmask_to_polygon(bitmask))
        elif annotation_type == AnnotationType.BITMASK:
            shape_dict_list = [bitmask_to_encord_dict(bitmask)]
            # FIXME: bug -
            print("WARNING - skipping bitmask prediction, due to bugged implementation")
            continue
        else:
            raise ValueError(f"Unknown annotation type for coco prediction import: {annotation_type}")

        append_object_to_list(
            object_list=predicted_data_unit.objects,
            annotation_type=annotation_type,
            shape_data_list=shape_dict_list,
            ontology_object=ontology_object,
            confidence=predict.score,
            object_hash=predict.encord_track_uuid,
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
