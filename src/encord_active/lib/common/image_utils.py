import json
from pathlib import Path
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


def draw_object(
    image: np.ndarray,
    row: pd.Series,
    mask_opacity: float = 0.5,
    color: Union[Color, str] = Color.PURPLE,
    with_box: bool = False,
):
    """
    The input should be a row from a LabelSchema (or descendants thereof).
    """
    isClosed = True
    thickness = get_polygon_thickness(image.shape[1])

    hex_color = color.value if isinstance(color, Color) else color
    _color: Tuple[int, ...] = hex_to_rgb(hex_color)
    _color_outline: Tuple[int, ...] = hex_to_rgb(hex_color, lighten=-0.5)

    box_only = not isinstance(row["rle"], str)
    if with_box or box_only:
        box = get_bbox_csv(row)
        image = cv2.polylines(image, [box], isClosed, _color, thickness, lineType=cv2.LINE_8)

    if box_only:
        return image

    mask = rle_to_binary_mask(eval(row["rle"]))

    # Draw contour line
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    image = cv2.polylines(image, contours, isClosed, _color_outline, thickness, lineType=cv2.LINE_8)

    # Fill polygon with opacity
    patch = np.zeros_like(image)
    mask_select = mask == 1
    patch[mask_select] = _color
    image[mask_select] = cv2.addWeighted(image, (1 - mask_opacity), patch, mask_opacity, 0)[mask_select]

    return image


def show_image_with_predictions_and_label(
    label: pd.Series,
    predictions: DataFrame[PredictionMatchSchema],
    data_dir: Path,
    label_color: Color = Color.RED,
    mask_opacity=0.5,
    class_colors: Optional[Dict[int, str]] = None,
):
    """
    Displays all predictions in the frame and the one label specified by the `label`
    argument. The label will be colored with the `label_color` provided and the
    predictions with a `purple` color unless `class_colors` are provided as a dict of
    (`class_id`, `<hex-color>`) pairs.

    :param label: The csv row of the false-negative label to display (from a LabelSchema).
    :param predictions: All the predictions on the same image with the samme predicted class (from a PredictionSchema).
    :param data_dir: The data directory of the project
    :param label_color: The hex color to use when drawing the prediction.
    :param class_colors: Dict of [class_id, hex_color] pairs.
    """
    class_colors = class_colors or {}
    image = load_or_fill_image(label, data_dir)

    for _, pred in predictions.iterrows():
        color = class_colors.get(pred["class_id"], Color.PURPLE)
        image = draw_object(image, pred, mask_opacity=mask_opacity, color=color)

    return draw_object(image, label, mask_opacity=mask_opacity, color=label_color, with_box=True)


def show_image_and_draw_polygons(
    row: Union[Series, str], data_dir: Path, draw_polygons: bool = True, skip_object_hash: bool = False
) -> np.ndarray:
    image = load_or_fill_image(row, data_dir)

    if not draw_polygons:
        return image

    is_closed = True
    thickness = get_polygon_thickness(image.shape[1])

    img_h, img_w = image.shape[:2]
    for color, geometry in get_geometries(row, img_h, img_w, data_dir, skip_object_hash=skip_object_hash):
        image = cv2.polylines(image, [geometry], is_closed, hex_to_rgb(color), thickness)

    return image


def load_or_fill_image(row: Union[pd.Series, str], data_dir: Path) -> np.ndarray:
    """
    Tries to read the infered image path. If not possible, generates a white image
    and indicates what the error seemd to be embedded in the image.
    :param row: A csv row from either a metric, a prediction, or a label csv file.
    :return: Numpy / cv2 image.
    """
    read_error = False
    key = __get_key(row)

    img_pth: Optional[Path] = key_to_image_path(key, data_dir)

    if img_pth and img_pth.is_file():
        try:
            image = cv2.imread(img_pth.as_posix())
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

    # Read not successful, so tell the user why
    error_text = "Image not found" if not img_pth else "File seems broken"

    _, du_hash, *_ = key.split("_")
    lr = json.loads(key_to_lr_path(key, data_dir).read_text(encoding="utf-8"))

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
        p = obj["polygon"]
        polygon = np.array([[p[str(i)]["x"] * img_w, p[str(i)]["y"] * img_h] for i in range(len(p))])
    elif obj["shape"] == ObjectShape.BOUNDING_BOX:
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
    row: Union[pd.Series, str], img_h: int, img_w: int, data_dir: Path, skip_object_hash: bool = False
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

    lr_pth = key_to_lr_path(key, data_dir)
    with lr_pth.open("r") as f:
        label_row = json.load(f)

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


def key_to_lr_path(key: str, data_dir: Path) -> Path:
    label_hash, *_ = key.split("_")
    return data_dir / label_hash / "label_row.json"


def key_to_image_path(key: str, data_dir: Path) -> Optional[Path]:
    """
    Infer image path from the identifier stored in the csv files.
    :param key: the row["identifier"] from a csv row
    :return: The associated image path if it exists or a path to a placeholder otherwise
    """
    label_hash, du_hash, frame, *_ = key.split("_")
    img_folder = data_dir / label_hash / "images"

    # check if it is a video frame
    frame_pth = next(img_folder.glob(f"{du_hash}_{int(frame)}.*"), None)
    if frame_pth is not None:
        return frame_pth
    return next(img_folder.glob(f"{du_hash}.*"), None)  # So this is an img_group image
