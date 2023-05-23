from typing import Optional, Union, Tuple, List

from torch import ByteTensor, BoolTensor
import torch
import math
from encord_active.analysis.metric import OneImageMetric, MetricDependencies, image_width, image_height


class ContrastMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident='contrast',
            dependencies={'brightness', 'area'},
            long_name='Contrast',
            short_desc='',
            long_desc='',
        )

    def calculate(self, deps: MetricDependencies, image: ByteTensor, mask: Optional[BoolTensor]) -> float:
        if mask is None:
            return torch.std(image).item() / 255
        else:
            # Masked standard deviation
            mask_mean = float(deps['img-brightness'])
            mask_count = float(deps['img-area'])
            mask_mean_delta = torch.masked_fill(image.float() - mask_mean, ~mask, 0)
            mask_mean_delta_sq = mask_mean_delta * mask_mean_delta
            mask_mean_delta_sq_sum = torch.sum(mask_mean_delta_sq)
            return math.sqrt(mask_mean_delta_sq_sum / mask_count)


class BrightnessMetric(OneImageMetric):

    def __init__(self) -> None:
        super().__init__(
            ident='brightness',
            dependencies={'area'},
            long_name='Brightness',
            short_desc='',
            long_desc='',
        )

    def calculate(self, deps: MetricDependencies, image: ByteTensor, mask: Optional[BoolTensor]) -> float:
        if mask is None:
            return torch.mean(image).item() / 255
        else:
            # Masked mean
            mask_count = float(deps['img-area'])
            mask_total = torch.sum(torch.masked_fill(image.long(), ~mask, 0)).item()
            return float(mask_total) / mask_count


class SharpnessMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident='sharpness',
            dependencies=set(),
            long_name='Sharpness',
            short_desc='',
            long_desc='',
        )

    def calculate(self, deps: MetricDependencies, image: ByteTensor, mask: Optional[BoolTensor]) -> float:
        # FIXME: Laplacian
        raise RuntimeError()


class AspectRatioMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident='aspect-ratio',
            dependencies=set(),
            long_name='Aspect Ratio',
            short_desc='',
            long_desc='',
            apply_to_objects=False,
            apply_to_classifications=False,
        )

    def calculate(self, deps: MetricDependencies, image: ByteTensor, mask: Optional[BoolTensor]) -> float:
        if mask is None:
            return float(image_width(image)) / float(image_height(image))
        else:
            shape0, shape1 = image.shape
            range_0: ByteTensor = torch.range(0, shape0).repeat(1, shape1)
            range_1: ByteTensor = torch.range(0, shape1).repeat(shape0, 1)
            mask_0_max = torch.masked_fill(range_0, ~mask, 0)
            mask_0_min = torch.masked_fill(range_0, ~mask, shape0)
            mask_1_max = torch.masked_fill(range_1, ~mask, 0)
            mask_1_min = torch.masked_fill(range_1, ~mask, shape1)
            width = torch.max(mask_0_max).item() - torch.min(mask_0_min).item()
            height = torch.max(mask_1_max).item() - torch.min(mask_1_min).item()
            return float(width) / float(height)


class AreaMetric(OneImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident='img-area',
            dependencies=set(),
            long_name='Area',
            short_desc='Area in pixels',
            long_desc='Area in pixels',
        )

    def calculate(self, deps: MetricDependencies, image: ByteTensor, mask: Optional[BoolTensor]) -> float:
        if mask is None:
            return float(image_width(image)) * float(image_height(image))
        else:
            return float(torch.sum(mask.long()).item())


HSVRange = Union[Tuple[int, int], List[Tuple[int, int]]]


class HSVMetric(OneImageMetric):
    def __init__(self,
                 color_name: str,
                 h_filter: HSVRange,
                 s_filter: HSVRange = (50, 255),
                 v_filter: HSVRange = (20, 255)) -> None:
        super().__init__(
            ident=color_name,
            dependencies=set(),
            long_name=f"{color_name} Values".title(),
            short_desc=f"Ranks images by how {color_name.lower()} the average value of the image is.",
            long_desc=f"Ranks images by how {color_name.lower()} the average value of the image is.",
        )

    def calculate(self, deps: MetricDependencies, image: ByteTensor, mask: Optional[BoolTensor]) -> float:
        # HSV Conversion TODO: implement this
        # FIXME: make hsv conversion an ephemeral embedding
        pass
