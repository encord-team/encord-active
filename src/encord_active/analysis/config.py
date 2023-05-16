from typing import List, Set, Union

from encord_active.analysis.embedding import ImageEmbedding
from encord_active.analysis.metric import ImageMetric
from encord_active.analysis.metrics.geometric.image_border_closeness import ImageBorderCloseness
from encord_active.analysis.metrics.heuristic.image_features import ContrastMetric, BrightnessMetric, SharpnessMetric, \
    AspectRatioMetric, AreaMetric
from encord_active.analysis.metrics.heuristic.object_count import ObjectCountMetric

""" List of all analysis passes to be executed by encord active"""
AllAnalysis: List[Union[ImageEmbedding, ImageMetric]] = [
    # Generate embeddings
    # TODO:
    # Image feature metrics
    AspectRatioMetric(),
    AreaMetric(),
    ContrastMetric(),
    BrightnessMetric(),
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
    AllAnalysisIdents.add(analysis.ident)
