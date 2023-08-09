import math
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from encord.objects import Object, OntologyStructure, Shape
from encord.utilities import label_utilities
from pydantic import BaseModel

from encord_active.db.models import (
    Project,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
)

from ...db.enums import AnnotationType
from ..util import append_object_to_list
from .op import ProjectImportSpec
from .util import get_data_uri


class CoCoFileInfo(BaseModel):
    description: str
    url: str
    version: Optional[str] = None
    year: str
    contributor: str
    date_created: str
    # Encord extra fields
    encord_title: Optional[str] = None


class CoCoLicenseInfo(BaseModel):
    url: str
    id: int
    name: str


class CoCoImageInfo(BaseModel):
    license: int
    file_name: str
    coco_url: Optional[str] = None
    height: int
    width: int
    date_captured: Optional[str] = None
    flickr_url: Optional[str] = None
    id: int
    # Encord extra fields


class CoCoBitmaskSegmentation(BaseModel):
    counts: List[int]


class CoCoAnnotationInfo(BaseModel):
    segmentation: Union[CoCoBitmaskSegmentation, List[List[float]]]
    area: float
    iscrowd: Literal[0, 1]
    image_id: int
    bbox: Tuple[float, float, float, float]
    category_id: int
    id: int
    track_id: Optional[int] = None
    keypoints: Optional[List[int]] = None
    num_keypoints: Optional[int] = None
    rotation: Optional[float] = None
    # Prediction / custom fields
    score: Optional[float] = None
    # Encord custom fields
    encord_track_uuid: Optional[str] = None
    ######


class CoCoCategoryInfo(BaseModel):
    supercategory: str
    id: int
    name: str
    # Encord extra fields
    feature_hash: Optional[float] = None


class CoCoFileSpec(BaseModel):
    info: CoCoFileInfo
    licenses: List[CoCoLicenseInfo]
    images: List[CoCoImageInfo]
    annotations: List[CoCoAnnotationInfo]
    categories: List[CoCoCategoryInfo]


def _get_image_url(image: CoCoImageInfo, images_dir_path: Optional[Path]) -> Union[str, Path]:
    if images_dir_path is not None:
        path = images_dir_path / image.file_name
        if path.exists():
            return path.expanduser().resolve()
    if image.coco_url is not None:
        return image.coco_url
    elif image.flickr_url is not None:
        return image.flickr_url
    raise ValueError(f"Failed to find image data for coco image in {images_dir_path}: {image}")


def _crop_box_to_image_size(x, y, w, h, image: CoCoImageInfo) -> Tuple[int, int, int, int]:
    if x > image.width:
        raise ValueError(f"x coordinate {x} of bounding box outside the image of width {image.width}")
    if y > image.height:
        raise ValueError(f"y coordinate {y} of bounding box outside the image of height {image.height}.")
    if x < 0:
        w += x
        x = 0
    if y < 0:
        h += y
        y = 0
    if x + w > image.width:
        w -= x + w - image.width
    if y + h > image.height:
        h -= y + h - image.height
    return x, y, w, h


def import_coco_annotation(
    annotation: CoCoAnnotationInfo, image: CoCoImageInfo, annotation_id_map: Dict[int, Tuple[AnnotationType, Object]]
) -> Tuple[List[dict], bool]:
    annotation_type, ontology_object = annotation_id_map[annotation.category_id]
    shape_data_list: List[Union[Dict[str, Dict[str, float]], Dict[str, float]]] = []
    if annotation.segmentation:
        if isinstance(annotation.segmentation, CoCoBitmaskSegmentation):
            # BITMASK: Not supported
            print("WARNING: BITMASK Coco Annotation found: this is not currently supported")
        else:
            for points in annotation.segmentation:
                if len(points) % 2 == 1:
                    raise ValueError(f"Cannot convert segmentation into polygon: {points}")
                shape_data_poly: Dict[str, Dict[str, float]] = {}
                for i in range(0, len(points), 2):
                    px = points[i]
                    py = points[i + 1]
                    pi = i // 2
                    shape_data_poly[str(pi)] = {"x": float(px / image.width), "y": float(py / image.height)}
                if annotation_type == AnnotationType.BITMASK:
                    raise ValueError("Bitmask conversion not implemented yet")
                else:
                    shape_data_list.append(shape_data_poly)
    elif len(annotation.bbox) == 4:
        x, y, w, h = _crop_box_to_image_size(*annotation.bbox, image=image)
        theta = float(annotation.rotation or 0.0)

        # Convert to polygon
        if annotation_type not in {AnnotationType.BOUNDING_BOX, AnnotationType.ROTATABLE_BOUNDING_BOX}:
            x2 = x + w
            y2 = y + h
            xm = x + (w / 2.0)
            ym = y + (h / 2.0)
            cs = math.cos(theta)
            sn = math.cos(theta)

            def _transform(xv: float, yv: float) -> Dict[str, float]:
                xv -= xm
                yv -= ym
                xr = (xv * cs) - (yv * sn)
                yr = (xv * sn) + (yv * cs)
                return {
                    "x": (xr + xm) / image.width,
                    "y": (yr + ym) / image.height,
                }

            shape_data_poly = {
                "0": _transform(x, y),
                "1": _transform(x, y2),
                "2": _transform(x2, y2),
                "3": _transform(x2, y),
            }
            if annotation_type == AnnotationType.BITMASK:
                raise ValueError("Bitmask conversion not implemented yet")
            else:
                shape_data_list.append(shape_data_poly)
        else:
            shape_data_bb: Dict[str, float] = {
                "x": float(x / image.width),
                "y": float(y / image.height),
                "w": float(w / image.width),
                "h": float(h / image.height),
            }
            if annotation is not AnnotationType.BOUNDING_BOX:
                shape_data_bb["theta"] = theta
            elif bool(annotation.rotation):
                raise ValueError("Inferred category id to bounding box, but rotation is requested")
            shape_data_list.append(shape_data_bb)
    else:
        raise ValueError(f"CoCo annotation with no segmentation & invalid bbox format: {annotation}")
    object_list: List[dict] = []
    append_object_to_list(
        object_list=object_list,
        annotation_type=annotation_type,
        shape_data_list=shape_data_list,
        ontology_object=ontology_object,
        confidence=annotation.score,
        object_hash=annotation.encord_track_uuid,
    )
    return object_list, True


def infer_coco_shape(coco_file: CoCoFileSpec) -> Dict[int, AnnotationType]:
    shapes_dict: Dict[int, AnnotationType] = {}
    for annotation in coco_file.annotations:
        # Set to the least general shape that is supported by both
        # BOUNDING_BOX -> ROTATABLE_BOUNDING_BOX -> POLYGON -> BITMASK
        annotation_type = shapes_dict.setdefault(annotation.category_id, AnnotationType.BOUNDING_BOX)
        if annotation.segmentation:
            if isinstance(annotation.segmentation, CoCoBitmaskSegmentation):
                print("WARNING: Bitmask CoCo Annotation Found, this is not currently supported")
                # FIXME: shapes_dict[annotation_type] = AnnotationType.BITMASK
            else:
                if annotation_type != AnnotationType.BITMASK:
                    shapes_dict[annotation.category_id] = AnnotationType.POLYGON
        elif len(annotation.bbox) == 4:
            if bool(annotation.rotation) and annotation_type == AnnotationType.BOUNDING_BOX:
                shapes_dict[annotation.category_id] = AnnotationType.ROTATABLE_BOUNDING_BOX
        else:
            raise ValueError(f"CoCo annotation with no segmentation & invalid bbox format: {annotation}")

    return shapes_dict


def import_coco_ontology(
    coco_file: CoCoFileSpec,
    shape_dict: Dict[int, AnnotationType],
) -> Tuple[dict, Dict[int, Tuple[AnnotationType, Object]]]:
    ontology_structure = OntologyStructure()
    id_mapping = {}
    for cat in coco_file.categories:
        shape = shape_dict.get(cat.id, AnnotationType.BITMASK)
        new_id = cat.id + 1  # NOTE: add one to the category id to avoid index 0
        name = cat.name
        obj = ontology_structure.add_object(name=name, shape=Shape(str(shape)), uid=new_id)
        id_mapping[cat.id] = shape, obj

    return ontology_structure.to_dict(), id_mapping


def import_coco(
    database_dir: Path,
    annotations_file_path: Path,
    images_dir_path: Optional[Path],
    store_data_locally: bool,
    store_symlinks: bool,
) -> ProjectImportSpec:
    coco_file: CoCoFileSpec = CoCoFileSpec.parse_file(annotations_file_path)
    project_hash = uuid.uuid4()
    dataset_hash = uuid.uuid4()
    dataset_title = ""

    # Generate 'ontology' for coco
    shape_dict = infer_coco_shape(coco_file)
    ontology, annotation_id_map = import_coco_ontology(
        coco_file,
        shape_dict=shape_dict,
    )
    project_data_list = []
    project_du_list = []

    # Transform images & annotations into encord spec.
    annotations_map: Dict[int, List[CoCoAnnotationInfo]] = {}
    for annotation in coco_file.annotations:
        annotations_map.setdefault(annotation.image_id, []).append(annotation)
    for image in coco_file.images:
        data_hash = uuid.uuid4()
        label_hash = uuid.uuid4()
        image_url_or_path = _get_image_url(image, images_dir_path)

        data_uri = get_data_uri(
            url_or_path=image_url_or_path,
            store_data_locally=store_data_locally,
            store_symlinks=store_symlinks,
            database_dir=database_dir,
        )

        # CoCo spec
        objects = []
        classifications = []
        for annotation in annotations_map[image.id]:
            coco_objects, is_object = import_coco_annotation(annotation, image, annotation_id_map)
            if is_object:
                objects.extend(coco_objects)
            else:
                classifications.extend(coco_objects)

        image_data_type = "png"
        if len(image.file_name.split(".")) > 1:
            image_data_type = image.file_name.split(".")[-1]

        # Save metadata
        project_data_list.append(
            ProjectDataMetadata(
                project_hash=project_hash,
                data_hash=data_hash,
                label_hash=label_hash,
                dataset_hash=dataset_hash,
                num_frames=1,
                frames_per_second=None,
                dataset_title=dataset_title,
                data_title=image.file_name,
                data_type="image",
                label_row_json=label_utilities.construct_answer_dictionaries(
                    {
                        "label_hash": str(label_hash),
                        "dataset_hash": str(dataset_hash),
                        "dataset_title": dataset_title,
                        "data_title": image.file_name,
                        "data_type": "image",
                        "data_units": {
                            str(data_hash): {
                                "data_hash": str(data_hash),
                                "data_title": image.file_name,
                                "data_type": f"image/{image_data_type}",
                                "data_sequence": 0,
                                "labels": {"objects": objects, "classifications": classifications},
                                "width": 512,
                                "height": 512,
                            }
                        },
                        "object_answers": {},
                        "classification_answers": {},
                        "object_actions": {},
                        "label_status": "NOT_LABELLED",
                    }
                ),
            )
        )
        project_du_list.append(
            ProjectDataUnitMetadata(
                project_hash=project_hash,
                du_hash=data_hash,
                frame=0,
                data_hash=data_hash,
                width=image.width,
                height=image.height,
                data_uri=data_uri,
                data_uri_is_video=False,
                objects=objects,
                classifications=classifications,
            )
        )

    # Transform into encord-alike project spec
    if coco_file.info.encord_title is not None:
        title = coco_file.info.encord_title
        description = coco_file.info.description
    else:
        title = coco_file.info.description
        description = ""
    return ProjectImportSpec(
        project=Project(
            project_hash=project_hash,
            project_name=title,
            project_description=description,
            project_remote_ssh_key_path=None,
            project_ontology=ontology,
        ),
        project_data_list=project_data_list,
        project_du_list=project_du_list,
    )
