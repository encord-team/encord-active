from typing import List

import streamlit as st

from encord_active.app.common.utils import setup_page
from encord_active.app.model_quality.prediction_type_builder import (
    ModelQualityPage,
    PredictionTypeBuilder,
)
from encord_active.app.model_quality.prediction_types.classification_type_builder import (
    ClassificationTypeBuilder,
)
from encord_active.app.model_quality.prediction_types.object_type_builder import (
    ObjectTypeBuilder,
)


def model_quality(page_mode: ModelQualityPage):
    def get_available_predictions() -> List[PredictionTypeBuilder]:
        builders: List[PredictionTypeBuilder] = [ClassificationTypeBuilder(), ObjectTypeBuilder()]
        return [b for b in builders if b.is_available()]

    def render():
        setup_page()

        available_predictions: List[PredictionTypeBuilder] = get_available_predictions()

        if not available_predictions:
            st.markdown("## No predictions imported into this project.")
            return

        if len(available_predictions) == 1:
            available_predictions[0].build(page_mode)
        else:
            tab_names = [m.title for m in available_predictions]
            for tab, builder in zip(st.tabs(tab_names), available_predictions):
                with tab:
                    builder.build(page_mode)

    return render
