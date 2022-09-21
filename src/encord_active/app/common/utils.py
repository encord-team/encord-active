import json
from json import JSONDecodeError
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import streamlit as st

import encord_active.app.common.state as state
from encord_active.app.common.colors import Color, hex_to_rgb
from encord_active.app.common.css import write_page_css
from encord_active.app.common.state import populate_session_state
from encord_active.app.db.merged_metrics import MergedMetrics
from encord_active.lib.common.utils import get_du_size


def set_page_config():
    project_root = Path(__file__).parents[1]
    favicon_pth = project_root / "assets" / "favicon-32x32.png"
    st.set_page_config(
        page_title="Encord Active",
        layout="wide",
        page_icon=favicon_pth.as_posix(),
    )


def setup_page():
    populate_session_state()
    write_page_css()


def load_json(json_file: Path) -> Optional[dict]:
    if not json_file.exists():
        return None

    with json_file.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except JSONDecodeError:
            return None


def load_or_fill_image(row: Union[pd.Series, str]) -> np.ndarray:
    """
    Tries to read the infered image path. If not possible, generates a white image
    and indicates what the error seemd to be embedded in the image.
    :param row: A csv row from either a metric, a prediction, or a label csv file.
    :return: Numpy / cv2 image.
    """

    read_error = False
    key = __get_key(row)

    img_pth: Optional[Path] = key_to_image_path(key)

    if img_pth and img_pth.is_file():
        try:
            image = cv2.imread(img_pth.as_posix())
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except cv2.error:
            pass

    # Read not successful, so tell the user why
    error_text = "Image not found" if not img_pth else "File seems broken"

    _, du_hash, *_ = key.split("_")
    lr = json.loads(key_to_lr_path(key).read_text(encoding="utf-8"))

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


def get_df_subset(df: pd.DataFrame, selected_metric: Optional[str]):
    if selected_metric not in df:
        return df

    max_val = float(df[selected_metric].max()) + np.finfo(float).eps.item()
    min_val = float(df[selected_metric].min())

    if max_val <= min_val:
        return df

    step = max(0.01, (max_val - min_val) // 100)
    start, end = st.slider("Choose quality", max_value=max_val, min_value=min_val, value=(min_val, max_val), step=step)
    subset = df[df[selected_metric].between(start, end)]

    return subset


def build_pagination(subset, n_cols, n_rows, selected_metric):
    n_items = n_cols * n_rows
    col1, col2 = st.columns(spec=[1, 4])

    with col1:
        sorting_order = st.selectbox("Sort samples within selected interval", ["Ascending", "Descending"])

    with col2:
        last = len(subset) // n_items + 1
        page_num = st.slider("Page", 1, last) if last > 1 else 1

    low_lim = (page_num - 1) * n_items
    high_lim = page_num * n_items

    sorted_subset = subset.sort_values(by=selected_metric, ascending=sorting_order == "Ascending")
    paginated_subset = sorted_subset[low_lim:high_lim]
    return paginated_subset


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

    if obj["shape"] == "polygon":
        p = obj["polygon"]
        polygon = np.array([[p[str(i)]["x"] * img_w, p[str(i)]["y"] * img_h] for i in range(len(p))])
    elif obj["shape"] == "bounding_box":
        b = obj["boundingBox"]
        polygon = np.array(
            [
                [b["x"] * img_w, b["y"] * img_h],
                [(b["x"] + b["w"]) * img_w, b["y"] * img_h],
                [(b["x"] + b["w"]) * img_w, (b["y"] + b["h"]) * img_h],
                [b["x"] * img_w, (b["y"] + b["h"]) * img_h],
            ]
        )
    else:
        return None

    polygon = polygon.reshape((-1, 1, 2)).astype(int)
    return obj.get("color", Color.PURPLE.value), polygon


def get_geometries(
    row: Union[pd.Series, str], img_h: int, img_w: int, skip_object_hash: bool = False
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

    lr_pth = key_to_lr_path(key)
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
            if obj["shape"] not in {"polygon", "bounding_box"}:
                continue
            geometries.append(__get_geometry(obj, img_h=img_h, img_w=img_w))

    valid_geometries = list(filter(None, geometries))
    return valid_geometries


def key_to_lr_path(key: str) -> Path:
    label_hash, *_ = key.split("_")
    return st.session_state.data_dir / label_hash / "label_row.json"


def key_to_image_path(key: str) -> Optional[Path]:
    """
    Infer image path from the identifier stored in the csv files.
    :param key: the row["identifier"] from a csv row
    :return: The associated image path if it exists or a path to a placeholder otherwise
    """
    label_hash, du_hash, frame, *_ = key.split("_")
    img_folder = st.session_state.data_dir / label_hash / "images"

    # check if it is a video frame
    frame_pth = next(img_folder.glob(f"{du_hash}_{int(frame)}.*"), None)
    if frame_pth is not None:
        return frame_pth
    return next(img_folder.glob(f"{du_hash}.*"), None)  # So this is an img_group image


def load_merged_df():
    if state.MERGED_DATAFRAME not in st.session_state:
        st.session_state[state.MERGED_DATAFRAME] = MergedMetrics().all()
