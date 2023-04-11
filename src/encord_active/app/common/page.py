from abc import ABC, abstractmethod

import streamlit as st

from encord_active.app.actions_page.export_filter import actions, show_update_stats
from encord_active.app.actions_page.versioning import version_form
from encord_active.app.common.components.sticky.sticky import sticky_header
from encord_active.app.common.components.tags.tag_creator import tag_creator
from encord_active.app.common.state import PageGridSettings, State, get_state
from encord_active.lib.common.image_utils import ObjectDrawingConfigurations


class Page(ABC):
    @property
    @abstractmethod
    def title(self) -> str:
        pass

    @abstractmethod
    def sidebar_options(self, *args, **kwargs):
        """
        Used to append object-level options to the sidebar.
        """
        pass

    def sidebar_options_classifications(self, *args, **kwargs):
        """
        Used to append classification-label options to the sidebar.
        """
        pass

    @abstractmethod
    def build(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    def __repr__(self):
        return f"{type(self).__name__}()"

    def display_settings(self, show_label_checkbox: bool = False):
        with sticky_header():
            with st.expander("Toolbox", expanded=True):
                view_tab, tag_tab, actions_tab, version_tab, options_tab, assistant_tab = st.tabs(
                    ["Filter", "Tag", "Action", "Version", "Options", "Assistant"]
                )

                with view_tab:
                    self.render_view_options()
                with tag_tab:
                    tag_creator()
                with actions_tab:
                    left, right = st.columns([5, 1])
                    with left:
                        actions()
                    with right:
                        filterd = get_state().filtering_state.merged_metrics
                        show_update_stats(filterd)
                with version_tab:
                    version_form()
                with options_tab:
                    self.render_common_settings(show_label_checkbox)
                with assistant_tab:
                    from encord_active.app.common.components.assistant import (
                        render_assistant,
                    )

                    render_assistant()

    def render_view_options(self):
        pass

    def render_common_settings(self, show_label_checkbox: bool = False):
        col_default_max, row_default_max = 10, 5
        setting_columns = st.columns(3)
        global_state = get_state()

        with setting_columns[0]:
            get_state().page_grid_settings.columns = int(
                st.number_input(
                    "Columns",
                    min_value=2,
                    max_value=col_default_max,
                    value=PageGridSettings.columns,
                    help="Number of columns to show images in the main view",
                )
            )
            if show_label_checkbox:
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
                    value=PageGridSettings.rows,
                    help="Number of rows to show images in the main view",
                )
            )
            global_state.object_drawing_configurations.contour_width = int(
                st.slider(
                    "Object contour width",
                    min_value=1,
                    max_value=25,
                    value=ObjectDrawingConfigurations.contour_width,
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
                    value=ObjectDrawingConfigurations.opacity,
                    help="Object opacities.",
                    disabled=not global_state.object_drawing_configurations.draw_objects,
                )
            )
