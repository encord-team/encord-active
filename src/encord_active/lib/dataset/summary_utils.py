import json
from pathlib import Path
from typing import Tuple

import numpy as np
from encord.constants.enums import DataType

from encord_active.lib.dataset.outliers import (
    MetricOutlierInfo,
    MetricsSeverity,
    MetricWithDistanceSchema,
    Severity,
    get_iqr_outliers,
)
from encord_active.lib.metrics.utils import MetricData, load_metric_dataframe

_COLUMNS = MetricWithDistanceSchema


def get_all_image_sizes(project_folder: Path) -> np.ndarray:
    image_sizes = []
    for label_row in (project_folder / "data").iterdir():
        if (label_row / "label_row.json").exists():
            label_row_meta = json.loads((label_row / "label_row.json").read_text(encoding="utf-8"))
            if label_row_meta["data_type"] in [DataType.IMAGE.value, DataType.IMG_GROUP.value]:
                for data_unit in label_row_meta["data_units"].values():
                    image_sizes.append([data_unit["width"], data_unit["height"]])

    return np.array(image_sizes)


def get_median_value_of_2d_array(array: np.ndarray) -> np.ndarray:
    """
    This function calculates the median value based on the product of the two dimension.
    For example, if they are image width and height, median dimensions corresponds to median average
    """
    product = array[:, 0] * array[:, 1]
    product_sorted = np.sort(product)
    median_value = product_sorted[product_sorted.size // 2]
    item_index = np.where(product == median_value)
    return array[item_index[0][0], :]


def get_all_annotation_numbers(project_folder: Path) -> Tuple[int, int]:
    """
    returns (number of classification label, number of object label)
    """
    classification_label_counter = 0
    object_label_counter = 0

    for label_row in (project_folder / "data").iterdir():
        if (label_row / "label_row.json").exists():
            label_row_meta = json.loads((label_row / "label_row.json").read_text(encoding="utf-8"))
            if label_row_meta["data_type"] in [DataType.IMAGE.value, DataType.IMG_GROUP.value]:
                for data_unit in label_row_meta["data_units"].values():
                    object_label_counter += len(data_unit["labels"]["objects"])
                    classification_label_counter += len(data_unit["labels"]["classifications"])

    return classification_label_counter, object_label_counter


def get_metric_summary(metrics: list[MetricData]) -> MetricsSeverity:
    metric_severity = MetricsSeverity()
    total_unique_severe_outliers = set()
    total_unique_moderate_outliers = set()

    for metric in metrics:
        original_df = load_metric_dataframe(metric, normalize=False)
        res = get_iqr_outliers(original_df)

        if not res:
            continue

        df, iqr_outliers = res

        df_dict = df.to_dict("records")
        for row in df_dict:
            if row[_COLUMNS.outliers_status] == Severity.severe:
                total_unique_severe_outliers.add(row[_COLUMNS.identifier])
            elif row[_COLUMNS.outliers_status] == Severity.moderate:
                total_unique_moderate_outliers.add(row[_COLUMNS.identifier])

        metric_severity.metrics.append(MetricOutlierInfo(metric=metric, df=df, iqr_outliers=iqr_outliers))

    metric_severity.total_unique_severe_outliers = len(total_unique_severe_outliers)
    metric_severity.total_unique_moderate_outliers = len(total_unique_moderate_outliers)

    return metric_severity
