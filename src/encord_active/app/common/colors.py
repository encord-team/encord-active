from enum import Enum
from typing import Tuple

import cv2
import numpy as np


class Color(Enum):
    PURPLE = "#5658dd"  # Encord purple
    GREEN = "#46D202"
    RED = "#FD0A5A"
    LITE_PURPLE = "#d449c1"
    PINK = "#ff5794"
    ORANGE = "#ff8a6a"
    LITE_ORANGE = "#ffc357"
    YELLOW = "#ffde00"
    BLUE = "#0075ef"


def hex_to_rgb(value: str, lighten=0.0) -> Tuple[int, ...]:
    """
    "#000011" -> (0, 0, 255)
    :param value: the hex color.
    :return: the rgb values.
    """
    value = value.lstrip("#")
    lv = len(value)
    color = np.array([int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3)], dtype=np.uint8)

    color_hsv = cv2.cvtColor(color.reshape((1, 1, 3)), cv2.COLOR_RGB2HSV)
    color_hsv[..., -1] = max(0, min(255, int(color_hsv[..., -1] * (1 + lighten))))

    color_rgb = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2RGB)
    return tuple(color_rgb.squeeze().tolist())
