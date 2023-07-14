import uuid
from typing import Dict, Tuple, Union, List, Optional, Type, Literal

from pydantic import BaseModel

from encord_active.db.metrics import MetricDefinition, DataMetrics, AnnotationMetrics
from encord_active.db.models import ProjectDataAnalyticsReduced, ProjectAnnotationAnalyticsReduced, \
    ProjectTaggedDataUnit, ProjectTaggedAnnotation, ProjectDataAnalytics, ProjectPredictionAnalytics, \
    ProjectAnnotationAnalytics


class Embedding2DFilter(BaseModel):
    reduction_hash: uuid.UUID
    min: Tuple[float, float]
    max: Tuple[float, float]


class DomainSearchFilters(BaseModel):
    metrics: Dict[str, Tuple[Union[int, float], Union[int, float]]]
    enums: Dict[str, List[str]]
    reduction: Optional[Embedding2DFilter]
    tags: Optional[List[str]]


class SearchFilters(BaseModel):
    data: Optional[DomainSearchFilters]
    annotation: Optional[DomainSearchFilters]


AnalyticsTable = Type[Union[ProjectDataAnalytics, ProjectAnnotationAnalytics, ProjectPredictionAnalytics]]
ReductionTable = Type[Union[ProjectDataAnalyticsReduced, ProjectAnnotationAnalyticsReduced]]
TagTable = Type[Union[ProjectTaggedDataUnit, ProjectTaggedAnnotation]]
TableJoinLiterals = List[Literal["project_hash", "prediction_hash", "du_hash", "frame", "object_hash"]]


class DomainTables(BaseModel):
    analytics: AnalyticsTable
    reduction: ReductionTable
    tag: TagTable
    join: TableJoinLiterals
    metrics: Dict[str, MetricDefinition]
    enums: Dict[str, dict]


class Tables(BaseModel):
    data: DomainTables
    annotation: Optional[DomainTables]


_DOMAIN_ANNOTATION = DomainTables(
    analytics=ProjectAnnotationAnalytics,
    reduction=ProjectAnnotationAnalyticsReduced,
    tag=ProjectTaggedAnnotation,
    join=["project_hash", "du_hash", "frame", "object_hash"],
    metrics=AnnotationMetrics,
    enums={}, # FIXME:
)
_DOMAIN_DATA = DomainTables(
    analytics=ProjectDataAnalytics,
    reduction=ProjectDataAnalyticsReduced,
    tag=ProjectTaggedDataUnit,
    join=["project_hash", "du_hash", "frame"],
    metrics=DataMetrics,
    enums={}, # FIXME:
)

TABLES_DATA = Tables(
    data=_DOMAIN_DATA,
    annotation=None,
)
TABLES_ANNOTATION = Tables(
    data=_DOMAIN_DATA,
    annotation=_DOMAIN_ANNOTATION,
)

TABLES_PREDICTION_TP_FP = Tables(
    data=_DOMAIN_DATA,
    annotation=None,  # FIXME: this is incorrect (join prediction against project hash is harder)
                      # FIXME: change prediction_hash & project_hash to be more hardcoded!!
                      # FIXME: store project_hash in join table for (all-prediction) filters & behavioural.
)
