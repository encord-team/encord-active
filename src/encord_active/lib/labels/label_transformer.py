import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import List, NamedTuple, Optional, Tuple, Union

import numpy as np
from encord.objects.classification import Classification
from encord.objects.common import NestableOption, RadioAttribute, Shape
from encord.objects.ontology_object import Object
from encord.ontology import OntologyStructure
from encord.project import LabelRow
from pydantic import BaseModel

from encord_active.lib.common.module_loading import load_module
from encord_active.lib.db.predictions import BoundingBox
from encord_active.lib.encord.utils import (
    Point,
    make_classification_dict_and_answer_dict,
    make_object_dict,
)


class Label(BaseModel):
    class_: str


class ClassificationLabel(Label):
    def __str__(self) -> str:
        return f"ClassificationLabel({self.class_})"


class BoundingBoxLabel(BaseModel):
    class_: str
    bounding_box: BoundingBox

    def __str__(self) -> str:
        return f"BounndingBoxLabel({self.class_}, {self.bounding_box})"


@dataclass
class PolygonLabel:
    class_: str
    polygon: np.ndarray
    """numpy arrays need to be an numpy array of relative (x, y) coordinates with shape [N, 2].
    For example: np.array([[0.1, 0.1], [0.1, 0.2], [0.2, 0.2], [0.2, 0.1]])
    """

    def __str__(self) -> str:
        n = self.polygon.shape[0]
        return f"PolygonLabel({self.class_}, [{n}, 2])"


LabelOptions = Union[BoundingBoxLabel, ClassificationLabel, PolygonLabel]


class DataLabel(NamedTuple):
    abs_data_path: Path
    label: LabelOptions


class LabelTransformer:
    """
    Extend the `LabelTransformer` class to transform your custom labels into labels that will work with Encord Active.
    You will have to implement the `from_custom_labels` function in order to do so.
    """

    def from_custom_labels(self, label_files: List[Path], data_files: List[Path]) -> List[DataLabel]:
        """
        Define the transform between your custom labels and Encord Actives LabelOptions.

        Args:
            label_files: The result of your `--label-glob` arguments.
            data_files: The result of your `--data-glob` arguments.

        Returns:
            The list of all the labels to import.

        """
        ...


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


class TransformerResult(NamedTuple):
    name: str
    transformer: LabelTransformer


def load_transformers_from_module(
    module_path: Union[str, Path],
) -> Optional[List[TransformerResult]]:
    mod = load_module(module_path)

    cls_members = inspect.getmembers(mod, inspect.isclass)
    label_transformers = []
    for cls_name, cls_obj in cls_members:
        if issubclass(cls_obj, LabelTransformer) and cls_obj != LabelTransformer:
            label_transformers.append(TransformerResult(cls_name, cls_obj()))
    return label_transformers
