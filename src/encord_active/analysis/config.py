from typing import List, Set

from encord_active.analysis.base import BaseEvaluation
from encord_active.analysis.metrics.geometric.image_border_closeness import ImageBorderCloseness
from encord_active.analysis.metrics.heuristic.image_features import ContrastMetric, BrightnessMetric, SharpnessMetric, \
    AspectRatioMetric, AreaMetric
from encord_active.analysis.metrics.heuristic.object_count import ObjectCountMetric

""" List of all analysis passes to be executed by encord active"""
AllAnalysis: List[BaseEvaluation] = [
    # Generate embeddings
    # TODO:
    # Image feature metrics
    AspectRatioMetric(),
    AreaMetric(),
    BrightnessMetric(),
    ContrastMetric(),
    SharpnessMetric(),
    # Geometric metrics
    ImageBorderCloseness(),
    # Heuristic metrics
    ObjectCountMetric(),
]

# Generate list of all analysis idents
AllAnalysisIdents: Set[str] = set()
for analysis in AllAnalysis:
    if analysis.ident in AllAnalysisIdents:
        raise RuntimeError(f"Duplicate analysis ident: {analysis.ident}")
    for dependency in analysis.dependencies:
        if dependency not in AllAnalysisIdents:
            raise RuntimeError(f"Dependency {dependency} for {analysis.ident} is not evaluated earlier in the ordering")
    AllAnalysisIdents.add(analysis.ident)
