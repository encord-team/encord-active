from abc import ABC, abstractmethod

import streamlit as st
from screeninfo import Monitor, ScreenInfoError, get_monitors

import encord_active.app.common.state as state
from encord_active.lib.common.metric import EmbeddingType


class Page(ABC):
    @property
    @abstractmethod
    def title(self) -> str:
        pass

    @abstractmethod
    def sidebar_options(self, *args, **kwargs):
        """
        Used to append options to the sidebar.
        """
        pass

    @abstractmethod
    def build(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def __repr__(self):
        return f"{type(self).__name__}()"

    @staticmethod
    def row_col_settings_in_sidebar():
        m_max = Monitor(height=0, width=0, x=0, y=0)
        try:
            monitors = get_monitors()
        except ScreenInfoError:
            monitors = []

        for m in monitors:
            if m.width * m.height > m_max.width * m_max.height:
                m_max = m

        col_default_max, row_default_max = 10, 4
        default_mv_column_num = min(m_max.width // 200, col_default_max) or 4
        default_mv_row_num = min(m_max.height // 200, row_default_max) or 4
        default_knn_num = 8

        with st.expander("Settings"):
            if state.MAIN_VIEW_COLUMN_NUM not in st.session_state:
                st.session_state[state.MAIN_VIEW_COLUMN_NUM] = default_mv_column_num

            if state.MAIN_VIEW_ROW_NUM not in st.session_state:
                st.session_state[state.MAIN_VIEW_ROW_NUM] = default_mv_row_num

            if state.K_NEAREST_NUM not in st.session_state:
                st.session_state[state.K_NEAREST_NUM] = default_knn_num

            with st.form(key="settings_from"):
                st.checkbox(
                    "Metric normalization",
                    key=state.NORMALIZATION_STATUS,
                    help="If checked, score values will be normalized between 0 and 1. Otherwise, \
                        original values will be shown.",
                )

                setting_columns = st.columns(2)

                setting_columns[0].number_input(
                    "Columns",
                    min_value=2,
                    max_value=col_default_max,
                    value=st.session_state[state.MAIN_VIEW_COLUMN_NUM],
                    key=state.MAIN_VIEW_COLUMN_NUM,
                    help="Number of columns to show images in the main view",
                )

                setting_columns[1].number_input(
                    "Rows",
                    min_value=1,
                    max_value=row_default_max,
                    value=st.session_state[state.MAIN_VIEW_ROW_NUM],
                    key=state.MAIN_VIEW_ROW_NUM,
                    help="Number of rows to show images in the main view",
                )

                if state.DATA_PAGE_METRIC in st.session_state.keys():
                    if st.session_state[state.DATA_PAGE_METRIC].meta.get(
                        "embedding_type", EmbeddingType.NONE.value
                    ) in [
                        EmbeddingType.CLASSIFICATION.value,
                        EmbeddingType.OBJECT.value,
                    ]:
                        st.number_input(
                            "k nearest",
                            min_value=4,
                            max_value=20,
                            key=state.K_NEAREST_NUM,
                            help="Number of the most similar images to show",
                        )
                st.form_submit_button(label="Apply")
