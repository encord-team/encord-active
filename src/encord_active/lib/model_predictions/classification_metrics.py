from pandera.typing import DataFrame

from encord_active.lib.model_predictions.reader import (
    ClassificationLabelSchema,
    ClassificationPredictionMatchSchema,
    ClassificationPredictionSchema,
)


def match_predictions_and_labels(
    model_predictions: DataFrame[ClassificationPredictionSchema], labels: DataFrame[ClassificationLabelSchema]
) -> DataFrame[ClassificationPredictionMatchSchema]:
    model_predictions = model_predictions.copy()
    labels = labels.copy()

    model_predictions[ClassificationPredictionMatchSchema.is_true_positive] = (
        model_predictions[ClassificationPredictionMatchSchema.class_id]
        .eq(labels[ClassificationLabelSchema.class_id])
        .astype(float)
    )

    return model_predictions.pipe(ClassificationPredictionMatchSchema)
