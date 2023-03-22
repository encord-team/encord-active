import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from pandas import Series
from pandera.typing import DataFrame
from shapely.affinity import rotate
from shapely.geometry import Polygon

from encord_active.lib.common.colors import Color, hex_to_rgb
from encord_active.lib.common.utils import get_du_size, rle_to_binary_mask
from encord_active.lib.db.predictions import BoundingBox
from encord_active.lib.labels.object import ObjectShape
from encord_active.lib.model_predictions.reader import PredictionMatchSchema
from encord_active.lib.project import (
    DataUnitStructure,
    LabelRowStructure,
    ProjectFileStructure,
)


@dataclass
class ObjectDrawingConfigurations:
    draw_objects: bool = True
    contour_width: int = 3
    opacity: float = 0.3


def get_polygon_thickness(img_w: int):
    t = max(1, int(img_w / 500))
    return t


def get_bbox_csv(row: pd.Series) -> np.ndarray:
    """
    Used to get a bounding box "polygon" for plotting.
    The input should be a row from a LabelSchema (or descendants thereof).
    """
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape((-1, 1, 2)).astype(int)


def draw_object_with_background_color(
    image: np.ndarray,
    geometry: Union[np.ndarray, list],
    color: Union[Color, str],
    draw_configurations: ObjectDrawingConfigurations,
):
    # if len(geometry) != 1:
    if isinstance(geometry, np.ndarray):
        if geometry.ndim == 2:
            geometry = geometry.reshape(-1, 1, 2)
        geometry = [geometry]

    """
        img_w_poly = cv2.fillPoly(image.copy(), [geometry], hex_to_rgb(color, lighten=0.5))
        image = cv2.addWeighted(image, 1 - draw_configurations.opacity, img_w_poly, draw_configurations.opacity, 1.0)
        image = cv2.polylines(image, [geometry], is_closed, hex_to_rgb(color), draw_configurations.contour_width)
    """

    hex_color = color.value if isinstance(color, Color) else color
    img_w_poly = cv2.fillPoly(image.copy(), geometry, hex_to_rgb(hex_color, lighten=0.5))
    image = cv2.addWeighted(image, 1 - draw_configurations.opacity, img_w_poly, draw_configurations.opacity, 1.0)
    return cv2.polylines(image, geometry, True, hex_to_rgb(hex_color), draw_configurations.contour_width)


def draw_object(
    image: np.ndarray,
    row: pd.Series,
    draw_configuration: Optional[ObjectDrawingConfigurations] = None,
    color: Union[Color, str] = Color.PURPLE,
    with_box: bool = False,
):
    """
    The input should be a row from a LabelSchema (or descendants thereof).
    """
    isClosed = True
    _color = hex_to_rgb(color.value if isinstance(color, Color) else color)
    if draw_configuration is None:
        draw_configuration = ObjectDrawingConfigurations()

    box_only = not isinstance(row["rle"], str)
    box = get_bbox_csv(row)

    if box_only:
        return draw_object_with_background_color(image, box, color, draw_configuration)

    if with_box:
        image = cv2.polylines(image, [box], isClosed, _color, draw_configuration.contour_width, lineType=cv2.LINE_8)

    mask = rle_to_binary_mask(eval(row["rle"]))
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    return draw_object_with_background_color(image, contours, color, draw_configuration)


def show_image_with_predictions_and_label(
    label: pd.Series,
    predictions: DataFrame[PredictionMatchSchema],
    project_file_structure: ProjectFileStructure,
    draw_configurations: Optional[ObjectDrawingConfigurations] = None,
    label_color: Color = Color.RED,
    class_colors: Optional[Dict[int, str]] = None,
):
    """
    Displays all predictions in the frame and the one label specified by the `label`
    argument. The label will be colored with the `label_color` provided and the
    predictions with a `purple` color unless `class_colors` are provided as a dict of
    (`class_id`, `<hex-color>`) pairs.

    :param label: The csv row of the false-negative label to display (from a LabelSchema).
    :param predictions: All the predictions on the same image with the samme predicted class (from a PredictionSchema).
    :param project_file_structure: The directory of the project
    :param label_color: The hex color to use when drawing the prediction.
    :param class_colors: Dict of [class_id, hex_color] pairs.
    """
    class_colors = class_colors or {}
    image = load_or_fill_image(label, project_file_structure)

    for _, pred in predictions.iterrows():
        color = class_colors.get(pred["class_id"], Color.PURPLE)
        image = draw_object(image, pred, draw_configurations, color=color)

    return draw_object(image, label, draw_configuration=draw_configurations, color=label_color, with_box=True)


def show_image_and_draw_polygons(
    row: Union[Series, str],
    project_file_structure: ProjectFileStructure,
    draw_configurations: Optional[ObjectDrawingConfigurations] = None,
    skip_object_hash: bool = False,
) -> np.ndarray:
    image = load_or_fill_image(row, project_file_structure)

    if draw_configurations is None:
        draw_configurations = ObjectDrawingConfigurations()

    if draw_configurations.draw_objects:
        img_h, img_w = image.shape[:2]
        for color, geometry in get_geometries(
            row, img_h, img_w, project_file_structure, skip_object_hash=skip_object_hash
        ):
            image = draw_object_with_background_color(image, geometry, color, draw_configurations)
    return image


def load_or_fill_image(row: Union[pd.Series, str], project_file_structure: ProjectFileStructure) -> np.ndarray:
    """
    Tries to read the inferred image path. If not possible, generates a white image
    and indicates what the error seemed to be embedded in the image.
    :param row: A csv row from either a metric, a prediction, or a label csv file.
    :return: Numpy / cv2 image.
    """
    key = __get_key(row)

    img_du: Optional[DataUnitStructure] = key_to_data_unit(key, project_file_structure)

    if img_du and img_du.path.is_file():
        try:
            image = cv2.imread(img_du.path.as_posix())
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

    # Read not successful, so tell the user why
    error_text = "Image not found" if not img_du else "File seems broken"

    _, du_hash, *_ = key.split("_")
    # lr = json.loads(key_to_lr_path(key, project_file_structure).read_text(encoding="utf-8"))
    label_row_structure = key_to_label_row_structure(key, project_file_structure)
    lr = json.loads(label_row_structure.label_row_file.read_text())

    h, w = get_du_size(lr["data_units"].get(du_hash, {}), None) or (600, 900)

    image = np.ones((h, w, 3), dtype=np.uint8) * 255
    image[:4, :] = [255, 0, 0]
    image[-4:, :] = [255, 0, 0]
    image[:, :4] = [255, 0, 0]
    image[:, -4:] = [255, 0, 0]
    font = cv2.FONT_HERSHEY_SIMPLEX
    pos = int(0.05 * min(w, h))
    cv2.putText(image, error_text, (pos, 2 * pos), font, w / 900, hex_to_rgb("#999999"), 2, cv2.LINE_AA)

    return image


def __get_key(row: Union[pd.Series, str]):
    if isinstance(row, pd.Series):
        if "identifier" not in row:
            raise ValueError("A Series passed but the series doesn't contain 'identifier'")
        return str(row["identifier"])
    elif isinstance(row, str):
        return row
    else:
        raise Exception(f"Undefined row type {row}")


def __get_geometry(obj: dict, img_h: int, img_w: int) -> Optional[Tuple[str, np.ndarray]]:
    """
    Convert Encord object dictionary to polygon coordinates used to draw geometries
    with opencv.

    :param obj: the encord object dict
    :param w: the image width
    :param h: the image height
    :return: The polygon coordinates
    """

    if obj["shape"] == ObjectShape.POLYGON:
        p = obj.get("polygon", {})
        if not p:
            return None
        polygon = np.array([[p[str(i)]["x"] * img_w, p[str(i)]["y"] * img_h] for i in range(len(p))])
    elif obj["shape"] == ObjectShape.BOUNDING_BOX:
        bbox_dict = obj.get("boundingBox", {})
        if not bbox_dict:
            return None
        b = BoundingBox.parse_obj(obj["boundingBox"])
        polygon = np.array(__to_absolute_points(b, img_h, img_w))
    elif obj["shape"] == ObjectShape.ROTATABLE_BOUNDING_BOX:
        b = BoundingBox.parse_obj(obj["rotatableBoundingBox"])
        rotated_polygon = rotate(Polygon(__to_absolute_points(b, img_h, img_w)), b.theta)
        if not rotated_polygon or not rotated_polygon.exterior:
            return None
        polygon = np.array(list(rotated_polygon.exterior.coords))
    else:
        return None

    polygon = polygon.reshape((-1, 1, 2)).astype(int)
    return obj.get("color", Color.PURPLE.value), polygon


def __to_absolute_points(bounding_box: BoundingBox, height: int, width: int):
    return [
        [bounding_box.x * width, bounding_box.y * height],
        [(bounding_box.x + bounding_box.w) * width, bounding_box.y * height],
        [(bounding_box.x + bounding_box.w) * width, (bounding_box.y + bounding_box.h) * height],
        [bounding_box.x * width, (bounding_box.y + bounding_box.h) * height],
    ]


def get_geometries(
    row: Union[pd.Series, str],
    img_h: int,
    img_w: int,
    project_file_structure: ProjectFileStructure,
    skip_object_hash: bool = False,
) -> List[Tuple[str, np.ndarray]]:
    """
    Loads cached label row and computes geometries from the label row.
    If the ``identifier`` in the ``row`` contains an object hash, only that object will
    be drawn. If no object hash exists, all polygons / bboxes will be drawn.
    :param row: the pandas row of the selected csv file.
    :return: List of tuples of (hex color, polygon: [[x, y], ...])
    """
    key = __get_key(row)
    _, du_hash, frame, *remainder = key.split("_")

    label_row_structure = key_to_label_row_structure(key, project_file_structure)
    label_row = json.loads(label_row_structure.label_row_file.read_text())
    du = label_row["data_units"][du_hash]

    geometries = []
    objects = (
        du["labels"].get("objects", [])
        if "video" not in du["data_type"]
        else du["labels"][str(int(frame))].get("objects", [])
    )

    if remainder and not skip_object_hash:
        # Return specific geometries
        geometry_object_hashes = set(remainder)
        for obj in objects:
            if obj["objectHash"] in geometry_object_hashes:
                geometries.append(__get_geometry(obj, img_h=img_h, img_w=img_w))
    else:
        # Get all geometries
        for obj in objects:
            if obj["shape"] not in {ObjectShape.POLYGON, ObjectShape.BOUNDING_BOX, ObjectShape.ROTATABLE_BOUNDING_BOX}:
                continue
            geometries.append(__get_geometry(obj, img_h=img_h, img_w=img_w))

    valid_geometries = list(filter(None, geometries))
    return valid_geometries


def key_to_label_row_structure(key: str, project_file_structure: ProjectFileStructure) -> LabelRowStructure:
    label_hash, *_ = key.split("_")
    return project_file_structure.label_row_structure(label_hash)


def key_to_data_unit(key: str, project_file_structure: ProjectFileStructure) -> Optional[DataUnitStructure]:
    """
    Infer image path from the identifier stored in the csv files.
    :param key: the row["identifier"] from a csv row
    :return: The associated image path if it exists or a path to a placeholder otherwise
    """
    label_hash, du_hash, frame, *_ = key.split("_")
    label_row_structure = project_file_structure.label_row_structure(label_hash)

    # check if it is a video frame
    frame_du: Optional[DataUnitStructure] = next(label_row_structure.iter_data_unit(du_hash, int(frame)), None)
    if frame_du is not None:
        return frame_du
    return next(label_row_structure.iter_data_unit(du_hash), None)  # So this is an img_group image
