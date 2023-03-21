from pandera.typing import DataFrame

from encord_active.lib.model_predictions.reader import (
    ClassificationLabelSchema,
    ClassificationPredictionMatchSchema,
    ClassificationPredictionSchema,
)


def match_predictions_and_labels(
    model_predictions: DataFrame[ClassificationPredictionSchema], labels: DataFrame[ClassificationLabelSchema]
) -> DataFrame[ClassificationPredictionMatchSchema]:
    _model_predictions = model_predictions.copy()
    _labels = labels.copy()

    _model_predictions[ClassificationPredictionMatchSchema.is_true_positive] = (
        _model_predictions[ClassificationPredictionSchema.class_id]
        .eq(_labels[ClassificationLabelSchema.class_id])
        .astype(float)
    )
    _model_predictions[ClassificationPredictionMatchSchema.gt_class_id] = _labels[ClassificationLabelSchema.class_id]

    return _model_predictions.pipe(ClassificationPredictionMatchSchema)
