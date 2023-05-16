from torch import std, mean

from encord_active.analysis.context import ImageEvalContext
from encord_active.analysis.image import ImageContext
from encord_active.analysis.metric import ImageMetric


class ContrastMetric(ImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident='img-contrast',
            dependencies=set(),
            long_name='Image Contrast',
            short_desc='',
            long_desc='',
        )

    def calculate(self, context: ImageEvalContext, image: ImageContext) -> float:
        return std(image.as_tensor()).item() / 255


class BrightnessMetric(ImageMetric):

    def __init__(self) -> None:
        super().__init__(
            ident='img-brightness',
            dependencies=set(),
            long_name='Image Brightness',
            short_desc='',
            long_desc='',
        )

    def calculate(self, context: ImageEvalContext, image: ImageContext) -> float:
        return mean(image.as_tensor()).item() / 255


class SharpnessMetric(ImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident='img-sharpness',
            dependencies=set(),
            long_name='Image Sharpness',
            short_desc='',
            long_desc='',
        )

    def calculate(self, context: ImageEvalContext, image: ImageContext) -> float:
        # FIXME: Laplacian
        raise RuntimeError()


class AspectRatioMetric(ImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident='img-aspect-ratio',
            dependencies=set(),
            long_name='Image Aspect Ratio',
            short_desc='',
            long_desc='',
        )

    def calculate(self, context: ImageEvalContext, image: ImageContext) -> float:
        return float(image.img_width()) / float(image.img_height())


class AreaMetric(ImageMetric):
    def __init__(self) -> None:
        super().__init__(
            ident='img-area',
            dependencies=set(),
            long_name='Image Area',
            short_desc='',
            long_desc='',
        )

    def calculate(self, context: ImageEvalContext, image: ImageContext) -> float:
        return float(image.img_width()) * float(image.img_height())
