from enum import Enum
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import streamlit as st
from natsort import natsorted
from pandas import Series

from encord_active.app.common.colors import hex_to_rgb
from encord_active.app.common.metric import MetricData
from encord_active.app.common.utils import get_geometries, load_json, load_or_fill_image


class MetricType(Enum):
    DATA_QUALITY = "data_quality"
    LABEL_QUALITY = "label_quality"


def get_metric_operation_level(pth: Path) -> str:
    if not all([pth.exists(), pth.is_file(), pth.suffix == ".csv"]):
        return ""

    with pth.open("r", encoding="utf-8") as f:
        _ = f.readline()  # Header, which we don't care about
        csv_row = f.readline()  # Content line

    if not csv_row:  # Empty metric
        return ""

    key, _ = csv_row.split(",", 1)
    _, _, _, *object_hashes = key.split("_")
    return "O" if object_hashes else "F"


@st.experimental_memo
def load_available_metrics(metric_type_selection: MetricType) -> List[MetricData]:
    def criterion(x):
        return x is None if metric_type_selection == MetricType.DATA_QUALITY.value else x is not None

    metric_dir = st.session_state.metric_dir
    if not metric_dir.is_dir():
        return []

    paths = natsorted([p for p in metric_dir.iterdir() if p.suffix == ".csv"], key=lambda x: x.stem.split("_", 1)[1])
    levels = list(map(get_metric_operation_level, paths))

    make_name = lambda p: p.name.split("_", 1)[1].rsplit(".", 1)[0].replace("_", " ").title()
    names = [f"{make_name(p)}" for p, l in zip(paths, levels)]
    meta_data = [load_json(f.with_suffix(".meta.json")) for f in paths]

    out: List[MetricData] = []

    if not meta_data:
        return out

    for p, n, m, l in zip(paths, names, meta_data, levels):
        if m is None or not l or not criterion(m.get("annotation_type")):
            continue

        out.append(MetricData(name=n, path=p, meta=m, level=l))

    out = natsorted(out, key=lambda i: (i.level, i.name))  # type: ignore
    return out


def show_image_and_draw_polygons(row: Union[Series, str], draw_polygons: bool = True) -> np.ndarray:
    # === Read and annotate the image === #
    image = load_or_fill_image(row)

    # === Draw polygons / bboxes if available === #
    is_closed = True
    thickness = int(image.shape[1] / 150)

    img_h, img_w = image.shape[:2]
    if draw_polygons:
        for color, geometry in get_geometries(row, img_h=img_h, img_w=img_w):
            image = cv2.polylines(image, [geometry], is_closed, hex_to_rgb(color), thickness)

    return image
