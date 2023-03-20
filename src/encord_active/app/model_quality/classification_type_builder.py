import encord_active.lib.model_predictions.reader as reader
from encord_active.app.common.state import get_state
from encord_active.app.model_quality.prediction_type_builder import (
    ModelQualityPage,
    PredictionTypeBuilder,
)
from encord_active.lib.model_predictions.writer import MainPredictionType


class ClassificationTypeBuilder(PredictionTypeBuilder):
    name = "Classification"

    def _load_Data(self):
        pass

    def is_available(self) -> bool:
        return reader.check_model_prediction_availability(
            get_state().project_paths.predictions / MainPredictionType.CLASSIFICATION.value
        )

    def render(self, page_mode: ModelQualityPage):
        pass
