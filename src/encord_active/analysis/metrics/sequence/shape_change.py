from typing import Optional

import torch
from shapely.geometry import Polygon

from encord_active.analysis.metric import TemporalOneObjectMetric
from encord_active.analysis.types import (
    AnnotationMetadata,
    MetricDependencies,
    MetricResult,
)
from encord_active.db.metrics import MetricType
from encord_active.lib.common.utils import get_iou


class TemporalShapeChange(TemporalOneObjectMetric):
    def __init__(self):
        super().__init__(
            ident="metric_polygon_similarity",
            long_name="Track Shape Outlier",
            desc="",
            metric_type=MetricType.NORMAL,
        )

    def _insert_midpoints(self, poly: torch.Tensor, target_length: int) -> torch.Tensor:
        """Insert midpoints into edges of the polygon until its length matches the target_length."""

        while len(poly) < target_length:
            # Find the longest edge
            edge_lengths = ((poly - torch.roll(poly, -1, dims=0)) ** 2).sum(dim=1)
            longest_edge_idx = edge_lengths.argmax().item()

            # Compute the midpoint of the longest edge
            start_point = poly[longest_edge_idx]
            end_point = poly[(longest_edge_idx + 1) % len(poly)]
            midpoint = (start_point + end_point) / 2

            # Insert the midpoint into the polygon
            poly = torch.cat([poly[: longest_edge_idx + 1], midpoint.unsqueeze(0), poly[longest_edge_idx + 1 :]])

        return poly

    def _is_counterclockwise(self, poly: torch.Tensor) -> bool:
        shapely_polygon = Polygon(list(zip(poly[:, 0].numpy(), poly[:, 1].numpy())))
        return shapely_polygon.exterior.is_ccw

    def calculate(
        self,
        annotation: AnnotationMetadata,
        deps: MetricDependencies,
        prev_annotation: Optional[AnnotationMetadata],
        next_annotation: Optional[AnnotationMetadata],
    ) -> MetricResult:

        ##########################################################################################
        # STEPS
        # 1. Orient polygon points in same direction. For example, they should be both clockwise
        # 2. Interpolate the small polygon to have same number of points as the large one.
        # 3. Match their starting point.
        # 4. Calculate interpolation points one-by-one
        ##########################################################################################

        if (
            prev_annotation is None
            or next_annotation is None
            or prev_annotation.points is None
            or next_annotation.points is None
            or annotation.points is None
        ):
            return 1.0

        prev_polygon = prev_annotation.points.clone()
        next_polygon = next_annotation.points.clone()

        N1, _ = prev_polygon.shape
        N2, _ = next_polygon.shape

        # STEP-1
        if self._is_counterclockwise(prev_polygon) != self._is_counterclockwise(next_polygon):
            next_polygon = next_polygon.flip(0)

        # STEP-2
        if N1 < N2:
            prev_polygon = self._insert_midpoints(prev_polygon, N2)
        elif N1 > N2:
            next_polygon = self._insert_midpoints(next_polygon, N1)

        # STEP-3
        distances = ((next_polygon - prev_polygon[0]) ** 2).sum(dim=1)
        min_distance_index = distances.argmin().item()
        next_polygon = torch.roll(next_polygon, shifts=-min_distance_index, dims=0)

        # Interpolate coordinates
        interpolated_coords = 0.5 * prev_polygon + 0.5 * next_polygon

        return get_iou(
            Polygon(list(zip(annotation.points[:, 0].numpy(), annotation.points[:, 1].numpy()))),
            Polygon(list(zip(interpolated_coords[:, 0].numpy(), interpolated_coords[:, 1].numpy()))),
        )
