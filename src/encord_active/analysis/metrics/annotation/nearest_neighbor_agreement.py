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
        nearest_neighbours: NearestNeighbors = cast(NearestNeighbors, deps["derived_clip_nearest_cosine"])
        if len(nearest_neighbours.similarities) == 0:
            # No nearest neighbours for agreement, hence score of 1
            return 1.0

        feature_hash: str = cast(str, deps["feature_hash"])
        matches = [float(dep["feature_hash"] == feature_hash) for dep in nearest_neighbours.metric_deps]
        similarity_sum = sum(nearest_neighbours.similarities)
        if similarity_sum > 0.0:
            # Weight by similarities
            # Relatively higher values (higher similarity, lower distance) should contribute more to the total
            matches = [m * (s / similarity_sum) for m, s in zip(matches, nearest_neighbours.similarities)]
            return min(sum(matches), 1.0)
        else:
            return min(sum(matches) / len(matches), 1.0)
