from encord_active.analysis.metric import DerivedMetric, MetricDependencies
from encord_active.analysis.types import MetricResult


class NearestNeighborAgreement(DerivedMetric):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            ident="nn-argeement",
            dependencies={"nn-search"},  # TODO this needs to be correct
            long_name="Nearest Neighbor Agreement",
            desc="Proportion of the nearest neighbors that share the same classification.",
            apply_to_images=False,
        )

    def calculate(self, deps: MetricDependencies) -> MetricResult:
        # TODO make this work
        return 0
