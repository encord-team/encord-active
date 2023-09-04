from typing import Optional, cast

from encord_active.analysis.metric import DerivedMetric, MetricDependencies
from encord_active.analysis.types import (
    AnnotationMetadata,
    MetricResult,
    NearestNeighbors,
)
from encord_active.db.metrics import MetricType


class NearestNeighborAgreement(DerivedMetric):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            ident="metric_annotation_quality",
            long_name="Nearest Neighbor Agreement",
            desc="Proportion of the nearest neighbors that share the same classification.",
            metric_type=MetricType.NORMAL,
        )

    def calculate(self, deps: MetricDependencies, annotation: Optional[AnnotationMetadata]) -> MetricResult:
        if annotation is None:
            return None
        # FIXME: experiment with different filtering strategies & weighting strategies
        #  find one that works the best.
        # calculate
        nearest_neighbours: NearestNeighbors = cast(NearestNeighbors, deps["derived_clip_nearest"])
        if len(nearest_neighbours.similarities) == 0:
            # No nearest neighbours for agreement, hence score of 0
            # FIXME: what should the default score be?
            return 0.0

        feature_hash: str = cast(str, deps["feature_hash"])
        matches = [float(dep["feature_hash"] == feature_hash) for dep in nearest_neighbours.metric_deps]
        similarity_sum = sum(nearest_neighbours.similarities)
        if similarity_sum > 0.0:
            # Weight by similarities
            # Relatively smaller values should contribute more to the total
            raw_bias = [1.0 / max(s, 0.01) for s in nearest_neighbours.similarities]
            total_bias = sum(raw_bias)
            matches = [m * (t / total_bias) for m, t in zip(matches, raw_bias)]
            return sum(matches)
        else:
            return sum(matches) / len(matches)
