import datetime
import uuid
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pytz
from encord.objects import OntologyStructure
from encord.objects.classification import Classification
from encord.objects.common import NestableOption, RadioAttribute, Shape
from encord.objects.ontology_object import Object
from encord.orm.label_row import LabelRow
from encord.project_ontology.object_type import ObjectShape
from pydantic import BaseModel

from encord_active.db.models import Project
from encord_active.imports.local_files import get_data_uri
from encord_active.imports.project.op import ProjectImportSpec
from encord_active.imports.project.util import data_du_meta_for_local_image
from encord_active.imports.util import get_mimetype
from encord_active.public.label_transformer import (
    BoundingBoxLabel,
    ClassificationLabel,
    KeypointLabel,
    LabelOptions,
    LabelTransformer,
    Point,
    PolygonLabel,
)

GMT_TIMEZONE = pytz.timezone("GMT")
DATETIME_STRING_FORMAT = "%a, %d %b %Y %H:%M:%S %Z"
BBOX_KEYS = {"x", "y", "h", "w"}


def get_timestamp():
    now = datetime.datetime.now()
    new_timezone_timestamp = now.astimezone(GMT_TIMEZONE)
    return new_timezone_timestamp.strftime(DATETIME_STRING_FORMAT)


SimpleShapes = {ObjectShape.POLYGON, ObjectShape.POLYLINE, ObjectShape.KEY_POINT, ObjectShape.SKELETON}
BoxShapes = {ObjectShape.BOUNDING_BOX, ObjectShape.ROTATABLE_BOUNDING_BOX}


def lower_snake_case(s: str):
    return "_".join(s.lower().split())


def make_object_dict(
    ontology_object: Object,
    object_data: Union[Point, List[Point], Dict[str, float], None] = None,
    object_hash: Optional[str] = None,
) -> dict:
    """
    :type ontology_object: The ontology object to associate with the ``object_data``.
    :type object_data: The data to put in the object dictionary. This has to conform
        with the ``shape`` parameter defined in
        ``ontology_object["shape"]``.
        - ``shape == "point"``: For key-points, the object data should be
          a tuple with x, y coordinates as floats.
        - ``shape == "bounding_box"``: For bounding boxes, the
          ``object_data`` needs to be a dict with info:
          {"x": float, "y": float, "h": float, "w": float} specifying the
          top right corner of the box (x, y) and the height and width of
          the bounding box.
        - ``shape in ("polygon", "polyline")``: For polygons and
          polylines, the format is an iterable  of points:
          [(x, y), ...] specifying (ordered) points in the
          polygon/polyline.
          If ``object_hash`` is none, a new hash will be generated.
    :type object_hash: If you want the object to have the same id across frames (for
        videos only), you can specify the object hash, which need to be
        an eight-character hex string (e.g., use
        ``str(uuid.uuid4())[:8]`` or the ``objectHash`` from an
        associated object.
    :returns: An object dictionary conforming with the Encord label row data format.
    """
    if object_hash is None:
        object_hash = str(uuid.uuid4())[:8]

    timestamp: str = get_timestamp()
    shape = ObjectShape(ontology_object.shape.value)

    object_dict = {
        "name": ontology_object.name,
        "color": ontology_object.color,
        "value": lower_snake_case(ontology_object.name),
        "createdAt": timestamp,
        "createdBy": "robot@cord.tech",
        "confidence": 1,
        "objectHash": object_hash,
        "featureHash": ontology_object.feature_node_hash,
        "lastEditedAt": timestamp,
        "lastEditedBy": "robot@encord.com",
        "shape": shape.value,
        "manualAnnotation": False,
        "reviews": [],
    }

    if shape in [ObjectShape.POLYGON, ObjectShape.POLYLINE] and object_data:
        if not isinstance(object_data, list):
            raise ValueError(f"The `object_data` for {shape} should be a list of points.")

        object_dict[shape.value] = {
            str(i): {"x": round(x, 4), "y": round(y, 4)} for i, (x, y) in enumerate(object_data)
        }

    elif shape == ObjectShape.KEY_POINT:
        if not isinstance(object_data, tuple):
            raise ValueError(f"The `object_data` for {shape} should be a tuple.")
        if len(object_data) != 2:
            raise ValueError(f"The `object_data` for {shape} should have two coordinates.")
        if not isinstance(object_data[0], float):
            raise ValueError(f"The `object_data` for {shape} should contain floats.")

        object_dict[shape.value] = {"0": {"x": round(object_data[0], 4), "y": round(object_data[1], 4)}}

    elif shape in BoxShapes:
        if not isinstance(object_data, dict):
            raise ValueError(f"The `object_data` for {shape} should be a dictionary.")
        if len(BBOX_KEYS.intersection(object_data.keys())) != 4:
            raise ValueError(f"The `object_data` for {shape} should have keys {BBOX_KEYS}.")
        if not isinstance(object_data["x"], float):
            raise ValueError(f"The `object_data` for {shape} should float values.")

        box = {k: round(v, 4) for k, v in object_data.items()}
        if shape == ObjectShape.ROTATABLE_BOUNDING_BOX:
            if "theta" not in object_data:
                raise ValueError(f"The `object_data` for {shape} should contain a `theta` field.")

            object_dict["rotatableBoundingBox"] = {**box, "theta": object_data["theta"]}
        else:
            object_dict["boundingBox"] = box

    else:
        raise RuntimeError(f"Unknown shape: {shape}")

    return object_dict


def make_classification_dict_and_answer_dict(
    classification: Classification,
    attribute: RadioAttribute,
    option: NestableOption,
    classification_hash: Optional[str] = None,
):
    if classification_hash is None:
        classification_hash = str(uuid.uuid4())[:8]

    answers = [
        {
            "featureHash": option.feature_node_hash,
            "name": option.label,
            "value": option.value,
        }
    ]

    classification_dict = {
        "classificationHash": classification_hash,
        "confidence": 1,
        "createdAt": get_timestamp(),
        "createdBy": "robot@encord.com",
        "featureHash": classification.feature_node_hash,
        "manualAnnotation": False,
        "name": attribute.name,
        "value": lower_snake_case(attribute.name),
    }

    classification_answer = {
        "classificationHash": classification_hash,
        "classifications": [
            {
                "answers": answers,
                "featureHash": attribute.feature_node_hash,
                "manualAnnotation": False,
                "name": attribute.name,
                "value": lower_snake_case(attribute.name),
            }
        ],
    }

    return classification_dict, classification_answer


class LabelTransformerWrapper:
    ATTRIBUTE_NAME = "What is in the frame?"

    def __init__(
        self,
        ontology_structure: OntologyStructure,
        label_rows: List[LabelRow],
        label_transformer: Optional[LabelTransformer],
    ):
        self.ontology_structure = ontology_structure
        self.label_transformer = label_transformer

        self.file_to_data_unit_map = {
            Path(du["data_link"]).expanduser().absolute(): du
            for lr in label_rows
            for du in lr.get("data_units", {}).values()
        }

        self.du_to_lr_map = {du["data_hash"]: lr for lr in label_rows for du in lr.get("data_units", {}).values()}

    def _get_object_by_name(self, name: str, shape: Shape, create_when_missing=False) -> Object:
        obj = next((o for o in self.ontology_structure.objects if o.name == name and o.shape == shape), None)
        if obj is None:
            if not create_when_missing:
                raise ValueError(
                    f"Couldn't find object with name `{name}`. Consider setting the `create_when_missing` flag to True."
                )

            obj = self.ontology_structure.add_object(name, shape=shape)
        return obj

    def _get_radio_classification_by_name(
        self, label: ClassificationLabel, create_when_missing=False
    ) -> Tuple[Classification, RadioAttribute, NestableOption]:
        classification: Optional[Classification] = None
        attribute: Optional[RadioAttribute] = None
        option: Optional[NestableOption] = None

        for clf in self.ontology_structure.classifications:
            for attr in clf.attributes:
                if not isinstance(attr, RadioAttribute):
                    continue

                for opt in attr.options:
                    if not isinstance(opt, NestableOption):
                        continue

                    if opt.label == label.class_:
                        classification = clf
                        attribute = attr
                        option = opt
                        break

                if option is not None:
                    break

            if option is not None:
                break

        if classification is not None and attribute is not None and option is not None:
            return classification, attribute, option

        if not create_when_missing:
            raise ValueError(
                f"""Couldn't find radio classification with option `{label.class_}`.
Consider setting the `create_when_missing` flag to True.
This will make a (potentially new) classification question `{LabelTransformerWrapper.ATTRIBUTE_NAME} and add the option `{label.class_}`."""
            )

        classification = next(
            (
                c
                for c in self.ontology_structure.classifications
                if LabelTransformerWrapper.ATTRIBUTE_NAME
                in [a.name for a in c.attributes if isinstance(a, RadioAttribute)]
            ),
            None,
        )
        if classification is None:
            classification = self.ontology_structure.add_classification()

        attribute = next(
            (
                a
                for a in classification.attributes
                if a.name == LabelTransformerWrapper.ATTRIBUTE_NAME and isinstance(a, RadioAttribute)
            ),
            None,
        )
        if attribute is None:
            attribute = classification.add_attribute(RadioAttribute, LabelTransformerWrapper.ATTRIBUTE_NAME)

        option = attribute.add_option(label.class_)

        return classification, attribute, option

    def _add_classification_label(self, label: ClassificationLabel, label_row: dict, data_unit: dict):
        clf, attr, opt = self._get_radio_classification_by_name(label, create_when_missing=True)
        clf_obj, clf_answer = make_classification_dict_and_answer_dict(clf, attr, opt)
        data_unit.setdefault("labels", {}).setdefault("classifications", []).append(clf_obj)
        label_row.setdefault("classification_answers", {})[clf_obj["classificationHash"]] = clf_answer

    def _add_bounding_box_label(self, label: BoundingBoxLabel, data_unit: dict):
        shape = Shape.BOUNDING_BOX if label.bounding_box.theta == 0.0 else Shape.ROTATABLE_BOUNDING_BOX

        ont_obj = self._get_object_by_name(label.class_, shape=shape, create_when_missing=True)
        bbox_dict = label.bounding_box.dict()
        if label.bounding_box.theta == 0:
            del bbox_dict["theta"]

        bbox_obj = make_object_dict(ont_obj, bbox_dict)
        data_unit.setdefault("labels", {}).setdefault("objects", []).append(bbox_obj)

    def _add_keypoint_label(self, label: KeypointLabel, data_unit: dict):
        shape = Shape.POINT

        ont_obj = self._get_object_by_name(label.class_, shape=shape, create_when_missing=True)
        bbox_obj = make_object_dict(ont_obj, label.point)
        data_unit.setdefault("labels", {}).setdefault("objects", []).append(bbox_obj)

    def _add_polygon_label(self, label: PolygonLabel, data_unit: dict):
        ont_obj = self._get_object_by_name(label.class_, shape=Shape.POLYGON, create_when_missing=True)
        points = [Point(*r) for r in label.polygon]
        poly_obj = make_object_dict(ont_obj, points)
        data_unit.setdefault("labels", {}).setdefault("objects", []).append(poly_obj)

    def _add_label(self, label: LabelOptions, label_row: dict, data_unit: dict):
        if isinstance(label, ClassificationLabel):
            self._add_classification_label(label, label_row, data_unit)
        elif isinstance(label, BoundingBoxLabel):
            self._add_bounding_box_label(label, data_unit)
        elif isinstance(label, PolygonLabel):
            self._add_polygon_label(label, data_unit)
        elif isinstance(label, KeypointLabel):
            self._add_keypoint_label(label, data_unit)
        else:
            raise RuntimeError(f"Unsupported label type: {type(label)}")

    def add_labels(self, label_paths: List[Path], data_paths: List[Path]):
        if not self.label_transformer:
            return

        for data_label in self.label_transformer.from_custom_labels(label_paths, data_paths):
            du = self.file_to_data_unit_map.get(data_label.abs_data_path.expanduser().resolve(), None)
            if du is None:
                continue

            lr = self.du_to_lr_map.get(du["data_hash"], None)
            if lr is None:
                continue

            self._add_label(data_label.label, lr, du)


class NoFilesFoundError(Exception):
    """Exception raised when searching for files yielded an empty result."""

    def __init__(self):
        super().__init__("Couldn't find any files to import from the given specifications.")


class GlobResult(BaseModel):
    matched: List[Path]
    excluded: List[Path]


def file_glob(root: Path, glob: List[str], images_only: bool = False) -> GlobResult:
    files = list(chain(*[root.glob(g) for g in glob]))

    if not len(files):
        raise NoFilesFoundError()

    if images_only:
        matches = []
        excluded = []
        for file in files:
            if "image" in get_mimetype(file):
                matches.append(file)
            else:
                excluded.append(file)
        return GlobResult(matched=matches, excluded=excluded)

    return GlobResult(matched=files, excluded=[])


def import_label_transformer(
    database_dir: Path,
    files: List[Path],
    project_name: str,
    symlinks: bool,
    label_transformer: Optional[LabelTransformer],
    label_paths: List[Path],
) -> ProjectImportSpec:
    ontology_structure = OntologyStructure()
    project_hash = uuid.uuid4()
    dataset_hash = uuid.uuid4()
    transformer_timestamp = datetime.datetime.now()

    path_to_data_hash: Dict[Path, uuid.UUID] = {file: uuid.uuid4() for file in files}
    data_hash_to_label_rows: Dict[uuid.UUID, LabelRow] = {
        data_hash: LabelRow(
            {
                "data_hash": str(dataset_hash),
                "label_hash": str(uuid.uuid4()),
                "data_units": {
                    str(data_hash): {
                        "data_hash": str(data_hash),
                        "data_link": path.as_posix(),
                    }
                },
                "object_answers": {},
                "classification_answers": {},
            }
        )
        for path, data_hash in path_to_data_hash.items()
    }

    # Import labels
    if label_transformer is not None:
        transformer = LabelTransformerWrapper(
            ontology_structure, list(data_hash_to_label_rows.values()), label_transformer
        )
        transformer.add_labels(label_paths or [], data_paths=files)
    elif len(label_paths) > 0:
        raise RuntimeError("Labels imported but no label transformer is selected")

    project_data_list = []
    project_du_list = []

    for file in files:
        data_hash = path_to_data_hash[file]
        label_row = data_hash_to_label_rows[data_hash]
        data_unit = label_row["data_units"][str(data_hash)]
        labels = data_unit.get("labels", {})
        data_uri = get_data_uri(
            url_or_path=file,
            store_data_locally=True,
            store_symlinks=symlinks,
            database_dir=database_dir,
        )
        data_meta, du_meta = data_du_meta_for_local_image(
            database_dir=database_dir,
            project_hash=project_hash,
            dataset_hash=dataset_hash,
            dataset_title=project_name,
            data_hash=data_hash,
            timestamp=transformer_timestamp,
            data_title=file.name,
            data_uri=data_uri,
            width=None,
            height=None,
            objects=labels.get("objects", []),
            classifications=labels.get("classifications", []),
            object_answers=label_row.get("object_answers", {}),
            classification_answers=label_row.get("classification_answers", {}),
        )
        project_data_list.append(data_meta)
        project_du_list.append(du_meta)

    return ProjectImportSpec(
        project=Project(
            project_hash=project_hash,
            name=project_name,
            description="",
            ontology=ontology_structure.to_dict(),
            remote=False,
        ),
        project_import_meta=None,
        project_data_list=project_data_list,
        project_du_list=project_du_list,
    )
