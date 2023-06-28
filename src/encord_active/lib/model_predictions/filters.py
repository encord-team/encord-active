import pandas as pd
from pandera.typing import DataFrame

from encord_active.lib.model_predictions.types import (
    ClassificationLabelSchema,
    ClassificationPredictionMatchSchema,
    ClassificationPredictionMatchSchemaWithClassNames,
    ClassificationPredictionSchema,
    PredictionMatchSchema,
)


def filter_labels_for_frames_wo_predictions(
    model_predictions: DataFrame[PredictionMatchSchema], sorted_labels: pd.DataFrame
):
    pred_keys = model_predictions["img_id"].unique()
    return sorted_labels[sorted_labels["img_id"].isin(pred_keys)]


def prediction_and_label_filtering_detection(
    selected_class_idx: dict,
    labels: pd.DataFrame,
    metrics: pd.DataFrame,
    model_pred: pd.DataFrame,
    precisions: pd.DataFrame,
):
    # Predictions
    class_idx = selected_class_idx
    row_selection = model_pred["class_id"].isin(set(map(int, class_idx.keys())))
    _model_pred = model_pred[row_selection].copy()

    # Labels
    row_selection = labels["class_id"].isin(set(map(int, class_idx.keys())))
    _labels = labels[row_selection]

    chosen_name_set = set(map(lambda x: x["name"], class_idx.values())).union({"Mean"})
    _metrics = metrics[metrics["class_name"].isin(chosen_name_set)]
    _precisions = precisions[precisions["class_name"].isin(chosen_name_set)]
    name_map = {int(k): v["name"] for k, v in class_idx.items()}
    _model_pred["class_name"] = _model_pred["class_id"].map(name_map)
    _labels["class_name"] = _labels["class_id"].map(name_map)
    return _labels, _metrics, _model_pred, _precisions


def prediction_and_label_filtering_classification(
    selected_class_idx: dict,
    all_class_idx: dict,
    labels: pd.DataFrame,
    predictions: pd.DataFrame,
    matched_model_predictions: DataFrame[ClassificationPredictionMatchSchema],
):
    class_idx = selected_class_idx
    new_index = max(list(map(int, all_class_idx.keys()))) + 1

    # Predictions
    _predictions = predictions.copy()
    _predictions.loc[
        ~_predictions[ClassificationPredictionSchema.class_id].isin(set(map(int, class_idx.keys()))),
        ClassificationPredictionSchema.class_id,
    ] = new_index

    # Labels
    _labels = labels.copy()
    _labels.loc[
        ~_labels[ClassificationLabelSchema.class_id].isin(set(map(int, class_idx.keys()))),
        ClassificationLabelSchema.class_id,
    ] = new_index

    name_map = {int(k): v["name"] for k, v in all_class_idx.items()}
    name_map[new_index] = "others"
    _predictions[ClassificationPredictionSchema.class_id] = _predictions[ClassificationPredictionSchema.class_id].map(
        name_map
    )
    _labels[ClassificationLabelSchema.class_id] = _labels[ClassificationLabelSchema.class_id].map(name_map)

    # matched predictions
    _matched_model_predictions = matched_model_predictions.copy()
    _matched_model_predictions = _matched_model_predictions[
        _matched_model_predictions[ClassificationPredictionMatchSchema.class_id].isin(set(map(int, class_idx.keys())))
    ]
    _matched_model_predictions[
        ClassificationPredictionMatchSchemaWithClassNames.class_name
    ] = _matched_model_predictions[ClassificationPredictionMatchSchema.class_id].map(name_map)

    _matched_model_predictions[
        ClassificationPredictionMatchSchemaWithClassNames.gt_class_name
    ] = _matched_model_predictions[ClassificationPredictionMatchSchema.gt_class_id].map(name_map)

    return _labels, _predictions, _matched_model_predictions.pipe(ClassificationPredictionMatchSchemaWithClassNames)
