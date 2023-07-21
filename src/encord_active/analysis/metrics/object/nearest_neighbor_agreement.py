from typing import Optional

from encord_active.analysis.metric import DerivedMetric, MetricDependencies
from encord_active.analysis.types import MetricResult
from encord_active.db.enums import AnnotationType


class NearestNeighborAgreement(DerivedMetric):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            ident="metric_annotation_quality",
            dependencies={"nn-search"},  # TODO this needs to be correct
            long_name="Nearest Neighbor Agreement",
            desc="Proportion of the nearest neighbors that share the same classification.",
        )

    def calculate(self, deps: MetricDependencies, annotation: Optional[AnnotationType]) -> MetricResult:
        # FIXME: - this is 'data metric' - check that this is where this should be stored.
        if annotation is None:
            return 0.0 # FIXME: stub value
        return None  # FIXME: implement properly
