from typing import Optional

from encord_active.analysis.metric import DerivedMetric, MetricDependencies
from encord_active.analysis.types import MetricResult, NearestNeighbors
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
        if annotation is None:
            return None
        # calculate
        nearest_neighbours: NearestNeighbors = deps["derived_clip_nearest"]
        feature_hash: str = deps["feature_hash"]
        matches = [
            float(dep["feature_hash"] == feature_hash)
            for dep in nearest_neighbours.metric_deps
        ]
        similarity_sum = sum(nearest_neighbours.similarities)
        if similarity_sum > 0.0:
            # Weight by similarities
            # Relatively smaller values should contribute more to the total
            raw_bias = [
                1.0 / max(s, 0.001)
                for s in nearest_neighbours.similarities
            ]
            total_bias = sum(raw_bias)
            matches = [
                m * (t / total_bias)
                for m, t in zip(matches, raw_bias)
            ]
        score = sum(matches) / len(matches)
        # print(f"Debug NN Agreement: {nearest_neighbours.similarities} / {matches} => {score}")
        return score
