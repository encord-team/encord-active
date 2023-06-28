import torch

from encord_active.analysis.metric import MetricDependencies, OneImageMetric
from encord_active.analysis.types import ImageTensor, MaskTensor, MetricResult


class HSVColorMetric(OneImageMetric):
    def __init__(self, color_name: str, hue_query: float) -> None:
        """
        Computes the average hue distance to hue_query across all pixels in the
        image. The Hue query should be a value in range [0; 1] where 0 is red,
        1/3 is green and 2/3 is blue.

        Args:
            color_name:
            hue_query:
        """
        super().__init__(
            ident=color_name,
            dependencies=set("hsv_image"),
            long_name=f"{color_name} Values".title(),
            desc=f"Ranks images by how {color_name.lower()} the average value of the image is.",
        )
        self.hue_query = hue_query * 2 * torch.pi

    def calculate(self, deps: MetricDependencies, image: ImageTensor, mask: MaskTensor | None) -> MetricResult:
        hsv_image = deps["hsv_image"]
        if not isinstance(hsv_image, torch.Tensor):
            raise ValueError("missing hsv image")

        if mask is None:
            hsv_pixels = hsv_image.reshape(3, -1)
        else:
            hsv_pixels = hsv_image[:, mask]

        hue_values = hsv_image[0]
        dists1 = (hue_values - self.hue_query) % 1.0
        dists2 = (self.hue_query - hue_values) % 1.0
        return torch.minimum(dists1, dists2).mean() * 2
