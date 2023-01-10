from abc import ABC, abstractmethod

import streamlit as st

from encord_active.app.common.state import State, get_state


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
        col_default_max, row_default_max = 10, 5

        with st.expander("Settings"):
            setting_columns = st.columns(3)

            get_state().page_grid_settings.columns = int(
                setting_columns[0].number_input(
                    "Columns",
                    min_value=2,
                    max_value=col_default_max,
                    value=State.page_grid_settings.columns,
                    help="Number of columns to show images in the main view",
                )
            )

            get_state().page_grid_settings.rows = int(
                setting_columns[1].number_input(
                    "Rows",
                    min_value=1,
                    max_value=row_default_max,
                    value=State.page_grid_settings.rows,
                    help="Number of rows to show images in the main view",
                )
            )

            get_state().similarities_count = int(
                setting_columns[2].number_input(
                    "Number of similar Images/Objects",
                    min_value=4,
                    max_value=20,
                    value=State.similarities_count,
                    help="Number of the most similar images to show",
                )
            )
