from encord_active.analysis.metric import DerivedMetric


class NearestNeighborAgreement(DerivedMetric):
    def __init__(
        self,
    ) -> None:
        super().__init__(
            ident: str,
            dependencies: set[str],
            long_name: str,
            desc: str,
            apply_to_images: bool = False,
            apply_to_objects: bool = True,
            apply_to_classifications: bool = True,
        )

    def calculate(self, deps: MetricDependencies) -> MetricResult:
        return 0
