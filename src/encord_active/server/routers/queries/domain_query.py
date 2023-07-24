import uuid
from typing import Dict, List, Literal, Optional, Type, Union

from pydantic import BaseModel

from encord_active.db.enums import AnnotationEnums, DataEnums, EnumDefinition
from encord_active.db.metrics import AnnotationMetrics, DataMetrics, MetricDefinition
from encord_active.db.models import (
    ProjectAnnotationAnalytics,
    ProjectAnnotationAnalyticsExtra,
    ProjectAnnotationAnalyticsReduced,
    ProjectDataAnalytics,
    ProjectDataAnalyticsExtra,
    ProjectDataAnalyticsReduced,
    ProjectPredictionAnalytics,
    ProjectPredictionAnalyticsExtra,
    ProjectPredictionAnalyticsFalseNegatives,
    ProjectPredictionAnalyticsReduced,
    ProjectTaggedAnnotation,
    ProjectTaggedDataUnit,
)

AnalyticsTable = Type[Union[ProjectDataAnalytics, ProjectAnnotationAnalytics, ProjectPredictionAnalytics]]
ReductionTable = Type[
    Union[ProjectDataAnalyticsReduced, ProjectAnnotationAnalyticsReduced, ProjectPredictionAnalyticsReduced]
]
MetadataTable = Type[Union[ProjectDataAnalyticsExtra, ProjectAnnotationAnalyticsExtra, ProjectPredictionAnalyticsExtra]]
TagTable = Type[Union[ProjectTaggedDataUnit, ProjectTaggedAnnotation]]
TableJoinLiterals = List[Literal["du_hash", "frame", "object_hash"]]

ProjectFilters = Dict[Literal["project_hash", "prediction_hash"], List[uuid.UUID]]


class DomainTables(BaseModel):
    analytics: AnalyticsTable
    metadata: MetadataTable
    reduction: ReductionTable
    tag: TagTable
    join: TableJoinLiterals
    metrics: Dict[str, MetricDefinition]
    enums: Dict[str, EnumDefinition]


class Tables(BaseModel):
    data: DomainTables
    annotation: Optional[DomainTables]


_DOMAIN_ANNOTATION = DomainTables(
    analytics=ProjectAnnotationAnalytics,
    metadata=ProjectAnnotationAnalyticsExtra,
    reduction=ProjectAnnotationAnalyticsReduced,
    tag=ProjectTaggedAnnotation,
    join=["du_hash", "frame", "object_hash"],
    metrics=AnnotationMetrics,
    enums=AnnotationEnums,
)
_DOMAIN_DATA = DomainTables(
    analytics=ProjectDataAnalytics,
    metadata=ProjectDataAnalyticsExtra,
    reduction=ProjectDataAnalyticsReduced,
    tag=ProjectTaggedDataUnit,
    join=["du_hash", "frame"],
    metrics=DataMetrics,
    enums=DataEnums,
)

_DOMAIN_PREDICTION_TP_FP = DomainTables(
    analytics=ProjectPredictionAnalytics,
    metadata=ProjectPredictionAnalyticsExtra,
    reduction=ProjectPredictionAnalyticsReduced,
    tag=ProjectTaggedDataUnit,  # FIXME: wrong table - define project tagged
    join=["du_hash", "frame", "object_hash"],
    metrics=AnnotationMetrics,
    enums=AnnotationEnums,
)

_DOMAIN_PREDICTION_FN_FIXME = DomainTables(  # FIXME: whole table is mostly wrong - we want to make decision on table
    # structure for false negatives
    analytics=ProjectPredictionAnalyticsFalseNegatives,
    metadata=ProjectAnnotationAnalyticsExtra,
    reduction=ProjectAnnotationAnalyticsReduced,
    tag=ProjectTaggedAnnotation,  # FIXME: change everything.
    join=["du_hash", "frame", "object_hash"],
    metrics=AnnotationMetrics,
    enums=AnnotationEnums,
)

TABLES_DATA = Tables(
    data=_DOMAIN_DATA,
    annotation=None,
)
TABLES_ANNOTATION = Tables(
    data=_DOMAIN_DATA,
    annotation=_DOMAIN_ANNOTATION,
)

# FIXME: this is incorrect (join prediction against project hash is harder)
# FIXME: change prediction_hash & project_hash to be more hardcoded!!
# FIXME: store project_hash in join table for (all-prediction) filters & behavioural.
TABLES_PREDICTION_TP_FP = Tables(
    data=_DOMAIN_DATA,
    # FIXME: join behaviour when switching between prediction_hash, annotation_hash
    #  needs to be optimized beyond requiring both for proper join behaviour
    annotation=_DOMAIN_PREDICTION_TP_FP,
)

# FIXME: we may aim to implement search over false negative metrics via join against annotations table
#  as we would otherwise be duplicating the requests.
TABLES_PREDICTION_FN = Tables(data=_DOMAIN_DATA, annotation=_DOMAIN_PREDICTION_FN_FIXME)
