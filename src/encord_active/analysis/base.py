from abc import ABC, abstractmethod
from typing import Dict, Optional

from pydantic import BaseModel

from encord_active.analysis.types import (
    AnnotationMetadata,
    ImageTensor,
    MetricDependencies,
    MetricResult,
)


class BaseFrameInput(BaseModel):
    image: ImageTensor
    image_deps: MetricDependencies
    # key is object_hash | classification_hash
    annotations: Dict[str, AnnotationMetadata]
    annotations_deps: Dict[str, MetricDependencies]
    # hash collision, if set the object or classification hash uniqueness constraint is
    # not held for this frame
    hash_collision: bool


class BaseFrameOutput(BaseModel):
    image: Optional[MetricResult]
    annotations: Dict[str, MetricResult]


class BaseEvaluation(ABC):
    """
    Base evaluations are the most abstract definitions of computations that need to happen.
    They can be both:

    - Quality Metrics that needs to be stored (The `BaseAnalysis` subclass)
    - Computations the need to happen for Quality Metrics to be able to share computations.
      For example, Embeddings, image transforms, etc.

    """

    def __init__(self, ident: str, dependencies: set[str]) -> None:
        self.ident = ident
        self.dependencies = dependencies

    @abstractmethod
    def raw_calculate(
        self,
        prev_frame: Optional[BaseFrameInput],
        frame: BaseFrameInput,
        next_frame: Optional[BaseFrameInput],
    ) -> BaseFrameOutput:
        """
        Base implementation of the raw_calculate method, this api should
        be considered unstable.
        """
        ...


class BaseAnalysis(BaseEvaluation, ABC):
    """
    The `BaseAnalysis` is all the metrics that needs to be stored in the
    database.
    """

    def __init__(self, ident: str, dependencies: set[str], long_name: str, desc: str) -> None:
        super().__init__(ident, dependencies)
        self.long_name = long_name
        self.desc = desc
