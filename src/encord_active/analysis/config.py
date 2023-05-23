from typing import List, Set, Optional, Dict
from dataclasses import dataclass

from torch import FloatTensor
from torch.types import Device

from encord_active.analysis.base import BaseEvaluation, TemporalBaseAnalysis
from encord_active.analysis.embedding import NearestImageEmbeddingQuery
from encord_active.analysis.embeddings.clip import ClipImgEmbedding
from encord_active.analysis.embeddings.hu_moment import HuMomentEmbeddings
from encord_active.analysis.metrics.geometric.image_border_closeness import ImageBorderCloseness
from encord_active.analysis.metrics.heuristic.image_features import ContrastMetric, BrightnessMetric, SharpnessMetric, \
    AspectRatioMetric, AreaMetric, HSVMetric
from encord_active.analysis.metrics.heuristic.object_count import ObjectCountMetric

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
4. Update object
    - internally implement as delete + add (as 1 transaction in db state)
5. Delete object
    - see new output
"""


@dataclass(frozen=True)
class InvalidatedKey:
    data_hash: str
    frame: int
    object_hash: Optional[str]


Invalidations = Set[InvalidatedKey]


@dataclass(frozen=True)
class MetricResult:
    value: float
    message: Optional[str]


@dataclass
class AnalysisResult:
    metrics: Dict[str, MetricResult]
    embeddings: Dict[str, FloatTensor]


@dataclass
class AnalysisResultWithObjects:
    image: AnalysisResult
    objects: Dict[str, AnalysisResult]


class AnalysisConfig:
    """
    Config for the complete analysis state.
    This should be generated once and re-used for all metric analysis
    Based upon the results.
    """

    def __init__(self,
                 analysis: List[BaseEvaluation],
                 derived_embeddings: List[NearestImageEmbeddingQuery],
                 derived_metrics: List[object]) -> None:
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

    def analysis_versions(self) -> Dict[str, int]:
        """
        Returns:
            The version for each exported metric or embedding,
            this value should be incremented whenever the code
            is changed in a way that would result in different results
            being returned if the function were to be re-executed.
            Exported features missing
        """
        ...

    def analyse_image(self, image_url: str) -> AnalysisResult:
        ...

    def analyse_video(self, video_url: str, start_timestamp: float,
                      start_frame: int, end_frame: int) -> AnalysisResult:
        ...

    def derive_embeddings(self, metric_db: None) -> None:
        pass

    def derive_metric(self, analysis_result: AnalysisResult) -> AnalysisResult:
        pass


def create_analysis(device: Device) -> AnalysisConfig:
    """ List of all analysis passes to be executed by encord active"""
    analysis = [
        # Generate embeddings
        ClipImgEmbedding(device, "clip", "ViT-B/32"),
        HuMomentEmbeddings("hu-moments"),
        # Image+object hybrid feature metrics
        AspectRatioMetric(),
        AreaMetric(),
        BrightnessMetric(),
        ContrastMetric(),
        SharpnessMetric(),
        HSVMetric("red", h_filter=[(0, 10), (170, 179)]),
        HSVMetric("green", h_filter=(35, 75)),
        HSVMetric("blue", h_filter=(90, 130)),
        # Geometric metrics
        ImageBorderCloseness(),
        # Heuristic metrics
        ObjectCountMetric(),
    ]
    derived_embeddings = [
        # Derive properties from embedding
        NearestImageEmbeddingQuery("clip-nearest", "clip"),
    ]
    derived_metrics = [
        # Metrics depending on derived embedding / metric properties ONLY
    ]
    return AnalysisConfig(
        analysis=analysis,
        derived_embeddings=derived_embeddings,
        derived_metrics=derived_metrics
    )


def verify_analysis(analysis_list: List[BaseEvaluation]) -> Set[str]:
    """Verify the config of the set of embedding and analysis passes to execute"""
    all_idents: Set[str] = set()
    for analysis in analysis_list:
        if analysis.ident in all_idents:
            raise RuntimeError(f"Duplicate analysis ident: {analysis.ident}")
        for dependency in analysis.dependencies:
            if dependency not in all_idents:
                raise RuntimeError(
                    f"Dependency {dependency} for {analysis.ident} is not evaluated earlier in the ordering")
        all_idents.add(analysis.ident)
    return all_idents


def split_analysis(analysis_list: List[BaseEvaluation]) -> List[List[BaseEvaluation]]:
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
