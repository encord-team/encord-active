from abc import ABC, abstractmethod

import streamlit as st

from encord_active.app.common.state import State, get_state
from encord_active.lib.common.image_utils import ObjectDrawingConfigurations
from encord_active.lib.metrics.utils import MetricScope


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
    def display_settings(metric_scope: MetricScope):
        col_default_max, row_default_max = 10, 5

        global_state = get_state()
        with st.expander("Settings"):
            setting_columns = st.columns(3)

            with setting_columns[0]:
                get_state().page_grid_settings.columns = int(
                    st.number_input(
                        "Columns",
                        min_value=2,
                        max_value=col_default_max,
                        value=State.page_grid_settings.columns,
                        help="Number of columns to show images in the main view",
                    )
                )
                if metric_scope == MetricScope.DATA_QUALITY:
                    st.markdown("")
                    st.markdown("")
                    global_state.object_drawing_configurations.draw_objects = st.checkbox(
                        "Show object labels",
                        value=ObjectDrawingConfigurations.draw_objects,
                        help="If checked, bounding boxes and polygons will be shown on the images.",
                    )
                else:
                    global_state.object_drawing_configurations.draw_objects = True

            with setting_columns[1]:
                global_state.page_grid_settings.rows = int(
                    st.number_input(
                        "Rows",
                        min_value=1,
                        max_value=row_default_max,
                        value=State.page_grid_settings.rows,
                        help="Number of rows to show images in the main view",
                    )
                )
                global_state.object_drawing_configurations.outline_width = int(
                    st.slider(
                        "Object outline widhth",
                        min_value=1,
                        max_value=25,
                        value=ObjectDrawingConfigurations.outline_width,
                        # value=State.object_drawing_configurations.outline_width,
                        help="Width of the objects outlines",
                        disabled=not global_state.object_drawing_configurations.draw_objects,
                    )
                )

            with setting_columns[2]:
                global_state.similarities_count = int(
                    st.number_input(
                        "Number of similar Images/Objects",
                        min_value=4,
                        max_value=20,
                        value=State.similarities_count,
                        help="Number of the most similar images to show",
                    )
                )

                global_state.object_drawing_configurations.opacity = float(
                    st.slider(
                        "Object opacity",
                        min_value=0.0,
                        max_value=1.0,
                        # value=State.object_drawing_configurations.opacity,
                        value=ObjectDrawingConfigurations.opacity,
                        help="Object opacities.",
                        disabled=not global_state.object_drawing_configurations.draw_objects,
                    )
                )
