from typing import NamedTuple, Optional, Tuple

import pandera as pa
from pandera.typing import DataFrame, Series

from encord_active.lib.metrics.utils import MetricSchema


class MetricWithDistanceSchema(MetricSchema):
    dist_to_iqr: Optional[Series[float]] = pa.Field()


class IqrOutliers(NamedTuple):
    n_moderate_outliers: int
    n_severe_outliers: int
    moderate_lb: float
    moderate_ub: float
    severe_lb: float
    severe_ub: float


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

    n_moderate_outliers = (
        ((severe_lb <= df[_COLUMNS.score]) & (df[_COLUMNS.score] < moderate_lb))
        | ((severe_ub >= df[_COLUMNS.score]) & (df[_COLUMNS.score] > moderate_ub))
    ).sum()

    n_severe_outliers = ((df[_COLUMNS.score] < severe_lb) | (df[_COLUMNS.score] > severe_ub)).sum()

    return (df, IqrOutliers(n_moderate_outliers, n_severe_outliers, moderate_lb, moderate_ub, severe_lb, severe_ub))
