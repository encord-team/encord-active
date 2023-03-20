from typing import List

import streamlit as st

from encord_active.app.common.components.tags.tag_creator import tag_creator
from encord_active.app.common.utils import setup_page
from encord_active.app.model_quality.classification_type_builder import (
    ClassificationTypeBuilder,
)
from encord_active.app.model_quality.object_type_builder import ObjectTypeBuilder
from encord_active.app.model_quality.prediction_type_builder import (
    ModelQualityPage,
    PredictionTypeBuilder,
)


def model_quality(page_mode: ModelQualityPage):
    def get_available_predictions() -> List[PredictionTypeBuilder]:
        builders = [ClassificationTypeBuilder(), ObjectTypeBuilder()]
        return [b for b in builders if b.is_available()]

    def render():
        setup_page()
        tag_creator()

        available_predictions: List[PredictionTypeBuilder] = get_available_predictions()

        if not available_predictions:
            st.markdown("## No predictions imported into this project.")
            pass
        st.write("2222")  # TODO delete

        tab_names = [m.name for m in available_predictions]
        tabs = st.tabs(tab_names)

        for tab, builder in zip(tabs, available_predictions):
            with tab:
                builder.build(page_mode)

    return render
