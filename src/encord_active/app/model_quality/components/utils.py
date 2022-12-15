from typing import Tuple, Union

import cv2
import numpy as np
import pandas as pd

from encord_active.app.common.colors import Color, hex_to_rgb
from encord_active.lib.common.utils import rle_to_binary_mask


def get_bbox(row: pd.Series):
    x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]).reshape((-1, 1, 2)).astype(int)


def draw_mask(row: pd.Series, image: np.ndarray, mask_opacity: float = 0.5, color: Union[Color, str] = Color.PURPLE):
    isClosed = True
    thickness = 2

    hex_color = color.value if isinstance(color, Color) else color
    _color: Tuple[int, ...] = hex_to_rgb(hex_color)
    _color_outline: Tuple[int, ...] = hex_to_rgb(hex_color, lighten=-0.5)
    if isinstance(row["rle"], str):
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
