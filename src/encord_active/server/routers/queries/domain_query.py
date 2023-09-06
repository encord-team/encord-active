import uuid
from typing import Dict, List, Literal, Type, Union, Tuple

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
TableJoinLiterals = List[Literal["du_hash", "frame", "annotation_hash"]]

ProjectFilters = Dict[Literal["project_hash", "prediction_hash"], List[uuid.UUID]]


class DomainTables(BaseModel):
    analytics: AnalyticsTable
    metadata: MetadataTable
    reduction: ReductionTable
    tag: TagTable
    join: TableJoinLiterals
    metrics: Dict[str, MetricDefinition]
    enums: Dict[str, EnumDefinition]
    domain: Literal["data", "annotation"]

    def select_args(
        self, table: Union[AnalyticsTable, MetadataTable, ReductionTable]
    ) -> Union[Tuple[uuid.UUID, int], Tuple[uuid.UUID, int, str]]:
        return tuple([  # type: ignore
            getattr(table, arg)
            for arg in self.join
        ])


class Tables(BaseModel):
    data: DomainTables
    annotation: DomainTables
    primary: DomainTables


_DOMAIN_ANNOTATION = DomainTables(
    analytics=ProjectAnnotationAnalytics,
    metadata=ProjectAnnotationAnalyticsExtra,
    reduction=ProjectAnnotationAnalyticsReduced,
    tag=ProjectTaggedAnnotation,
    join=["du_hash", "frame", "annotation_hash"],
    metrics=AnnotationMetrics,
    enums=AnnotationEnums,
    domain="annotation",
)
_DOMAIN_DATA = DomainTables(
    analytics=ProjectDataAnalytics,
    metadata=ProjectDataAnalyticsExtra,
    reduction=ProjectDataAnalyticsReduced,
    tag=ProjectTaggedDataUnit,
    join=["du_hash", "frame"],
    metrics=DataMetrics,
    enums=DataEnums,
    domain="data"
)

_DOMAIN_PREDICTION_TP_FP = DomainTables(
    analytics=ProjectPredictionAnalytics,
    metadata=ProjectPredictionAnalyticsExtra,
    reduction=ProjectPredictionAnalyticsReduced,
    tag=ProjectTaggedDataUnit,  # FIXME: wrong table - define project tagged
    join=["du_hash", "frame", "annotation_hash"],
    metrics=AnnotationMetrics,
    enums=AnnotationEnums,
    domain="annotation"
)

_DOMAIN_PREDICTION_FN_FIXME = DomainTables(  # FIXME: whole table is mostly wrong - we want to make decision on table
    # structure for false negatives
    analytics=ProjectPredictionAnalyticsFalseNegatives,
    metadata=ProjectAnnotationAnalyticsExtra,
    reduction=ProjectAnnotationAnalyticsReduced,
    tag=ProjectTaggedAnnotation,  # FIXME: change everything.
    join=["du_hash", "frame", "annotation_hash"],
    metrics=AnnotationMetrics,
    enums=AnnotationEnums,
    domain="annotation",
)

TABLES_DATA = Tables(
    data=_DOMAIN_DATA,
    annotation=_DOMAIN_DATA,
    primary=_DOMAIN_DATA,
)
TABLES_ANNOTATION = Tables(
    data=_DOMAIN_DATA,
    annotation=_DOMAIN_ANNOTATION,
    primary=_DOMAIN_ANNOTATION
)

# FIXME: this is incorrect (join prediction against project hash is harder)
# FIXME: change prediction_hash & project_hash to be more hardcoded!!
# FIXME: store project_hash in join table for (all-prediction) filters & behavioural.
TABLES_PREDICTION_TP_FP = Tables(
    data=_DOMAIN_DATA,
    # FIXME: join behaviour when switching between prediction_hash, annotation_hash
    #  needs to be optimized beyond requiring both for proper join behaviour
    annotation=_DOMAIN_PREDICTION_TP_FP,
    primary=_DOMAIN_PREDICTION_TP_FP,
)
