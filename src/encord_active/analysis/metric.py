from abc import ABC, abstractmethod
from typing import List, Tuple, Set

from encord_active.analysis.base import BaseAnalysis
from encord_active.analysis.context import ImageEvalContext
from encord_active.analysis.image import ImageContext


class ImageMetric(BaseAnalysis, ABC):
    def __init__(self, ident: str, dependencies: Set[str], long_name: str, short_desc: str, long_desc: str) -> None:
        super().__init__(ident=ident, dependencies=dependencies)
        self.long_name = long_name
        self.short_desc = short_desc
        self.long_desc = long_desc

    @abstractmethod
    def calculate(self, context: ImageEvalContext, image: ImageContext) -> float:
        ...


class VideoDeltaImageMetric(BaseAnalysis, ABC):
    def __init__(self, ident: str, dependencies: Set[str], long_name: str, short_desc: str, long_desc: str, prev_frame_count: int,
                 next_frame_count: int) -> None:
        super().__init__(ident=ident, dependencies=dependencies)
        self.long_name = long_name
        self.short_desc = short_desc
        self.long_desc = long_desc
        self.prev_frame_count = prev_frame_count
        self.next_frame_count = next_frame_count

    @abstractmethod
    def calculate(self,
                  context: ImageEvalContext,
                  image: ImageContext,
                  prev_frames: List[Tuple[ImageContext, ImageEvalContext]],
                  next_frames: List[Tuple[ImageContext, ImageEvalContext]]) -> float:
        """Calculate a metric with a dependence on previous and next frames for a video"""
        ...


class LabelMetric(BaseAnalysis, ABC):
    def __init__(self, ident: str, dependencies: Set[str], long_name: str, short_desc: str, long_desc: str) -> None:
        super().__init__(ident=ident, dependencies=dependencies)
        self.long_name = long_name
        self.short_desc = short_desc
        self.long_desc = long_desc

    @abstractmethod
    def calculate_object(self, context: ImageEvalContext, image: ImageContext, obj: dict) -> float:
        ...
