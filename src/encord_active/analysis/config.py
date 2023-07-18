import torch
from torch.types import Device

from encord_active.analysis.metrics.object.nearest_neighbor_agreement import (
    NearestNeighborAgreement,
)

from .base import BaseEvaluation
from .conversions.hsv import RGBToHSV
from .embedding import NearestImageEmbeddingQuery
from .embeddings.clip import ClipImgEmbedding
from .embeddings.hu_moment import HuMomentEmbeddings
from .metric import DerivedMetric
from .metrics.image.area import AreaMetric
from .metrics.image.aspect_ratio import AspectRatioMetric
from .metrics.image.brightness import BrightnessMetric
from .metrics.image.colors import HSVColorMetric
from .metrics.image.contrast import ContrastMetric
from .metrics.image.height import HeightMetric
from .metrics.image.random import RandomMetric
from .metrics.image.sharpness import SharpnessMetric
from .metrics.image.width import WidthMetric
from .metrics.object.count import ObjectCountMetric
from .metrics.object.distance_to_border import DistanceToBorderMetric
from .metrics.object.maximum_label_iou import MaximumLabelIOUMetric

"""
Operations to support tracking for:
1. Add new input (image/video/...):
    - calculate new data
    - invalidate nearby embeddings
2. Delete input
    - cascade delete data
    - invalidate nearby embeddings
3. Add object
    - see new input
    - calculate new data
    - invalidate nearby embeddings
4. Update object
    - internally implement as delete + add (as 1 transaction in db state)
5. Delete object
    - see new output
    - cascade delete data
    - invalidate nearby embeddings
"""


class AnalysisConfig:
    """
    Config for the complete analysis state.
    This should be generated once and re-used for all metric analysis
    Based upon the results.
    """

    def __init__(
        self,
        analysis: list[BaseEvaluation],
        derived_embeddings: list[NearestImageEmbeddingQuery],
        derived_metrics: list[DerivedMetric],
    ) -> None:
        """
        Args:
            analysis:
                List of pure analysis that only depend on the input state, or
                earlier analysis results in the list. This can generate
                temporary tensors, embeddings and metrics.
            derive_embeddings:
                List of embeddings based properties that are derived from input embeddings,
                all operations:
                    - Nearest Embedding
                    - Nearest N Embeddings (where N is hardcoded constant).
            derive_metrics:
                List of metrics that only take exported results from earlier analysis and
                derived_embedding results.
        """
        self.analysis = analysis
        self.derived_embeddings = derived_embeddings
        self.derived_metrics = derived_metrics

    def validate(self) -> None:
        """
        Asserts that the given analysis config is valid for execution.
        """
        ...

    def analysis_versions(self) -> dict[str, int]:
        """
        Returns:
            The version for each exported metric or embedding,
            this value should be incremented whenever the code
            is changed in a way that would result in different results
            being returned if the function were to be re-executed.
            Exported features missing
        """
        ...


def default_torch_device() -> Device:
    """Default torch device to use for the analysis config"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_analysis(device: Device) -> AnalysisConfig:
    """List of all analysis passes to be executed by encord active"""
    analysis: list[BaseEvaluation] = [
        # Generate embeddings
        ClipImgEmbedding(device, "clip", "ViT-B/32"),
        HuMomentEmbeddings("hu-moments"),
        # Data conversions
        RGBToHSV(),
        # Image+object hybrid feature metrics
        AspectRatioMetric(),
        AreaMetric(),
        BrightnessMetric(),
        ContrastMetric(),
        SharpnessMetric(),
        HSVColorMetric("red", hue_query=0.0),
        HSVColorMetric("green", hue_query=1 / 3.0),
        HSVColorMetric("blue", hue_query=2 / 3.0),
        HeightMetric(),
        WidthMetric(),
        RandomMetric(),
        # Geometric metrics
        MaximumLabelIOUMetric(),
        # Heuristic metrics
        DistanceToBorderMetric(),
        ObjectCountMetric(),
    ]
    derived_embeddings: list[NearestImageEmbeddingQuery] = [
        # Derive properties from embedding
        # TODO ☝️  The typing needs to be more general eventually
        NearestImageEmbeddingQuery("clip-nearest", "clip"),
    ]
    derived_metrics: list[DerivedMetric] = [
        # Metrics depending on derived embedding / metric properties ONLY
        NearestNeighborAgreement()
    ]
    return AnalysisConfig(analysis=analysis, derived_embeddings=derived_embeddings, derived_metrics=derived_metrics)


def verify_analysis(analysis_list: list[BaseEvaluation]) -> set[str]:
    """Verify the config of the set of embedding and analysis passes to execute"""
    all_idents: set[str] = set()
    for analysis in analysis_list:
        if analysis.ident in all_idents:
            raise RuntimeError(f"Duplicate analysis ident: {analysis.ident}")
        for dependency in analysis.dependencies:
            if dependency not in all_idents:
                raise RuntimeError(
                    f"Dependency {dependency} for {analysis.ident} is not evaluated earlier in the ordering"
                )
        all_idents.add(analysis.ident)
    return all_idents


def split_analysis(analysis_list: list[BaseEvaluation]) -> list[list[BaseEvaluation]]:
    complex_dependency_set = set()
    res = [[]]
    for analysis in analysis_list:
        if any(dependency in complex_dependency_set for dependency in analysis.dependencies):
            res.append([])
            complex_dependency_set = set()
        res[-1].append(analysis)
        if isinstance(analysis, NearestImageEmbeddingQuery) or isinstance(analysis, TemporalBaseAnalysis):
            complex_dependency_set.add(analysis.ident)

    return res
