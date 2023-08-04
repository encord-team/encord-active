from kornia.color import rgb_to_hsv as ko_rgb_to_hsv

from encord_active.analysis.types import HSVTensor, ImageTensor


def rgb_to_hsv(image: ImageTensor) -> HSVTensor:
    tensor = image.float() / 255
    return ko_rgb_to_hsv(tensor)
