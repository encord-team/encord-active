import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, List, NamedTuple, Optional, Union

import numpy as np
from pydantic import BaseModel, Field

from encord_active.lib.common.module_loading import load_module

RelativeFloat = Annotated[float, Field(ge=0, le=1)]
DegreeFloat = Annotated[float, Field(ge=0, le=360)]

__all__ = [
    "RelativeFloat",
    "DegreeFloat",
    "BaseModel",
    "Point",
    "Label",
    "ClassificationLabel",
    "BoundingBoxLabel",
    "KeypointLabel",
    "PolygonLabel",
    "LabelOptions",
    "DataLabel",
    "LabelTransformer",
]


class BoundingBox(BaseModel):
    x: RelativeFloat
    y: RelativeFloat
    h: RelativeFloat
    w: RelativeFloat
    theta: DegreeFloat = 0.0


class Point(NamedTuple):
    x: RelativeFloat
    y: RelativeFloat


class Label(BaseModel):
    class_: str


class ClassificationLabel(Label):
    def __str__(self) -> str:
        return f"ClassificationLabel({self.class_})"


class BoundingBoxLabel(Label):
    bounding_box: BoundingBox

    def __str__(self) -> str:
        return f"BounndingBoxLabel({self.class_}, {self.bounding_box})"


class KeypointLabel(Label):
    point: Point

    def __str__(self) -> str:
        return f"KeypointLabel({self.class_})"


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


LabelOptions = Union[BoundingBoxLabel, ClassificationLabel, PolygonLabel, KeypointLabel]


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
