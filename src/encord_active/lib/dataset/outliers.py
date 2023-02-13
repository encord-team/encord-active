from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple, Optional, Tuple

import pandas as pd
import pandera as pa
from pandera.typing import DataFrame, Series

from encord_active.lib.metrics.utils import MetricData, MetricSchema


class MetricWithDistanceSchema(MetricSchema):
    dist_to_iqr: Optional[Series[float]] = pa.Field(coerce=True)
    outliers_status: Optional[Series[str]] = pa.Field(coerce=True)


class IqrOutliers(NamedTuple):
    n_moderate_outliers: int
    n_severe_outliers: int
    moderate_lb: float
    moderate_ub: float
    severe_lb: float
    severe_ub: float


class MetricOutlierInfo(NamedTuple):
    metric: MetricData
    df: DataFrame[MetricWithDistanceSchema]
    iqr_outliers: IqrOutliers


@dataclass
class MetricsSeverity:
    metrics: dict[str, MetricOutlierInfo] = field(default_factory=dict)
    total_unique_moderate_outliers: Optional[int] = None
    total_unique_severe_outliers: Optional[int] = None


class Severity(str, Enum):
    severe = "Severe"
    moderate = "Moderate"
    low = "Low"


class AllMetricsOutlierSchema(pa.SchemaModel):
    metric_name: Series[str]
    total_severe_outliers: Series[int] = pa.Field(ge=0, coerce=True)
    total_moderate_outliers: Series[int] = pa.Field(ge=0, coerce=True)


_COLUMNS = MetricWithDistanceSchema


def get_iqr_outliers(
    input: DataFrame[MetricSchema],
) -> Optional[Tuple[DataFrame[MetricWithDistanceSchema], IqrOutliers]]:
    if input.empty:
        return None

    df = DataFrame[MetricWithDistanceSchema](input)

    moderate_iqr_scale = 1.5
    severe_iqr_scale = 2.5
    Q1 = df[_COLUMNS.score].quantile(0.25)
    Q3 = df[_COLUMNS.score].quantile(0.75)
    IQR = Q3 - Q1

    df[_COLUMNS.dist_to_iqr] = 0
    df.loc[df[_COLUMNS.score] > Q3, _COLUMNS.dist_to_iqr] = (df[_COLUMNS.score] - Q3).abs()
    df.loc[df[_COLUMNS.score] < Q1, _COLUMNS.dist_to_iqr] = (df[_COLUMNS.score] - Q1).abs()
    df.sort_values(by=_COLUMNS.dist_to_iqr, inplace=True, ascending=False)

    moderate_lb, moderate_ub = Q1 - moderate_iqr_scale * IQR, Q3 + moderate_iqr_scale * IQR
    severe_lb, severe_ub = Q1 - severe_iqr_scale * IQR, Q3 + severe_iqr_scale * IQR

    df[_COLUMNS.outliers_status] = Severity.low
    df.loc[
        ((severe_lb <= df[_COLUMNS.score]) & (df[_COLUMNS.score] < moderate_lb))
        | ((severe_ub >= df[_COLUMNS.score]) & (df[_COLUMNS.score] > moderate_ub)),
        _COLUMNS.outliers_status,
    ] = Severity.moderate
    df.loc[
        (df[_COLUMNS.score] < severe_lb) | (df[_COLUMNS.score] > severe_ub), _COLUMNS.outliers_status
    ] = Severity.severe

    n_moderate_outliers = (
        ((severe_lb <= df[_COLUMNS.score]) & (df[_COLUMNS.score] < moderate_lb))
        | ((severe_ub >= df[_COLUMNS.score]) & (df[_COLUMNS.score] > moderate_ub))
    ).sum()

    n_severe_outliers = ((df[_COLUMNS.score] < severe_lb) | (df[_COLUMNS.score] > severe_ub)).sum()

    return df, IqrOutliers(n_moderate_outliers, n_severe_outliers, moderate_lb, moderate_ub, severe_lb, severe_ub)


def get_all_metrics_outliers(metrics_data_summary: MetricsSeverity) -> DataFrame[AllMetricsOutlierSchema]:
    all_metrics_outliers = pd.DataFrame(
        columns=[
            AllMetricsOutlierSchema.metric_name,
            AllMetricsOutlierSchema.total_severe_outliers,
            AllMetricsOutlierSchema.total_moderate_outliers,
        ]
    )
    for item in metrics_data_summary.metrics.values():
        all_metrics_outliers = pd.concat(
            [
                all_metrics_outliers,
                pd.DataFrame(
                    {
                        AllMetricsOutlierSchema.metric_name: [item.metric.name],
                        AllMetricsOutlierSchema.total_severe_outliers: [item.iqr_outliers.n_severe_outliers],
                        AllMetricsOutlierSchema.total_moderate_outliers: [item.iqr_outliers.n_moderate_outliers],
                    }
                ),
            ],
            axis=0,
        )

    all_metrics_outliers.sort_values(by=[AllMetricsOutlierSchema.total_severe_outliers], ascending=False, inplace=True)

    return all_metrics_outliers.pipe(DataFrame[AllMetricsOutlierSchema])
