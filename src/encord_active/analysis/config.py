import os
from typing import Union

import torch

from .base import BaseEvaluation
from .conversions.grayscale import RGBToGray
from .conversions.hsv import RGBToHSV
from .conversions.laplacian import RGBToLaplacian
from .embedding import NearestImageEmbeddingQuery
from .embeddings.clip import ClipImgEmbedding
from .embeddings.hu_moment import HuMomentEmbeddings
from .metric import DerivedMetric, RandomSamplingQuery
from .metrics.active_learning.uniqueness import ImageUniqueness
from .metrics.annotation.count import ObjectCountMetric
from .metrics.annotation.density import ObjectDensityMetric
from .metrics.annotation.distance_to_border import DistanceToBorderMetric
from .metrics.annotation.maximum_label_iou import MaximumLabelIOUMetric
from .metrics.annotation.nearest_neighbor_agreement import NearestNeighborAgreement
from .metrics.image.area import AreaMetric, AreaRelativeMetric
from .metrics.image.aspect_ratio import AspectRatioMetric
from .metrics.image.brightness import BrightnessMetric
from .metrics.image.colors import HSVColorMetric
from .metrics.image.contrast import ContrastMetric
from .metrics.image.height import HeightMetric
from .metrics.image.random_value import RandomMetric
from .metrics.image.sharpness import SharpnessMetric
from .metrics.image.width import WidthMetric

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
        derived_embeddings: list[Union[NearestImageEmbeddingQuery, RandomSamplingQuery]],
        derived_metrics: list[DerivedMetric],
        device: torch.device,
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
        self.device = device

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


def default_torch_device() -> torch.device:
    """Default torch device to use for the analysis config"""
    torch_device_str = "cpu"
    # FIXME: currently CPU only due to GPU being too slow due to lack of batch support.
    #  keep this until batch functions are properly implemented.
    if torch.cuda.is_available() and os.environ.get("ACTIVE_ALLOW_TORCH_CUDA_BACKEND", "0") == "1":
        torch_device_str = "cuda"
    elif torch.backends.mps.is_available() and os.environ.get("ACTIVE_ALLOW_TORCH_MPS_BACKEND", "0") == "1":
        torch_device_str = "mps"
    return torch.device(torch_device_str)


def create_analysis(device: torch.device) -> AnalysisConfig:
    """List of all analysis passes to be executed by encord active"""
    analysis: list[BaseEvaluation] = [
        # Generate embeddings
        ClipImgEmbedding(device, "embedding_clip", "ViT-B/32"),
        HuMomentEmbeddings(),
        # FIXME: HuMomentEmbeddings("embedding_hu_moments"),
        # Image transformations
        RGBToLaplacian(),
        RGBToGray(),
        RGBToHSV(),
        # Unified Image or Annotation Feature Metrics
        AspectRatioMetric(),  # metric_aspect_ratio
        AreaMetric(),  # metric_area
        BrightnessMetric(),  # metric_brightness
        ContrastMetric(),  # metric_contrast
        SharpnessMetric(),  # metric_sharpness
        HSVColorMetric("red", hue_query=0.0),  # metric_red
        HSVColorMetric("green", hue_query=1 / 3.0),  # metric_green
        HSVColorMetric("blue", hue_query=2 / 3.0),  # metric_blue
        HeightMetric(),  # metric_height
        WidthMetric(),  # metric_width
        RandomMetric(),  # metric_random
        # Annotation only metrics
        AreaRelativeMetric(),  # metric_area_relative
        MaximumLabelIOUMetric(),  # metric_max_iou
        DistanceToBorderMetric(),  # metric_border_relative
        # Annotation based Frame Metrics
        ObjectCountMetric(),  # metric_object_count
        ObjectDensityMetric(),  # metric_object_density ("Frame object density")
        # ?????
        # FIXME: metric_image_difficulty ("Image Difficulty")  FIXME: KMeans!!??
        # FIXME: metric_label_inconsistent_class_or_track ("Inconsistent Object Classification and Track IDs")
        # FIXME: metric_label_shape_outlier ("Shape outlier detection")
        # FIXME: metric_seq_occlusion_detection (ALSO - MISSING COLUMN NAME, WILL NEED ALEMBIC MIGRATION)
        # Temporal metrics
        # FIXME: DISABLED TemporalShapeChange(),  # metric_label_poly_similarity ("Polygon Shape Similarity")
        # FIXME: DISABLED TemporalMissingObjectsAndWrongTracks(),  # metric_missing_or_broken_track ("Missing Objects and Broken Tracks")
    ]
    derived_embeddings: list[Union[NearestImageEmbeddingQuery, RandomSamplingQuery]] = [
        # Derive properties from embedding
        NearestImageEmbeddingQuery("derived_clip_nearest", "embedding_clip"),
    ]
    derived_metrics: list[DerivedMetric] = [
        # Metrics depending on derived embedding / metric properties ONLY
        NearestNeighborAgreement(),  # metric_annotation_quality
        ImageUniqueness(),  # metric_image_uniqueness ("Image Singularity")
    ]
    return AnalysisConfig(
        analysis=analysis,
        # FIXME?: derived_temporal (derived metrics that have access to analysis values in N frame range)
        derived_embeddings=derived_embeddings,
        derived_metrics=derived_metrics,
        device=device,
    )


def get_default_analysis():
    analysis = create_analysis(torch.device("cpu"))
