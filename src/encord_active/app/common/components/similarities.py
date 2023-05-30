from typing import Callable, Union

import numpy as np
from pandas import Series
from streamlit.delta_generator import DeltaGenerator

from encord_active.app.common.state import get_state
from encord_active.lib.common.image_utils import (
    load_or_fill_image,
    show_image_and_draw_polygons,
)
from encord_active.lib.embeddings.utils import SimilaritiesFinder
from encord_active.lib.project import ProjectFileStructure


def show_similarities(identifier: str, expander: DeltaGenerator, embedding_information: SimilaritiesFinder):
    nearest_images = embedding_information.get_similarities(identifier)

    division = get_state().page_grid_settings.columns
    column_id = 0
    st_columns = []

    for nearest_image in nearest_images:
        if column_id == 0:
            st_columns = expander.columns(division)

        load_image: Callable[[Union[Series, str], ProjectFileStructure], np.ndarray] = load_or_fill_image
        if embedding_information.has_annotations:
            load_image = show_image_and_draw_polygons

        image = load_image(nearest_image["key"], get_state().project_paths)

        st_columns[column_id].image(image)
        st_columns[column_id].write(f"Annotated as `{nearest_image['name']}`")
        column_id += 1
        column_id = column_id % division
