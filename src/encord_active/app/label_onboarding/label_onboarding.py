from enum import Enum
from pathlib import Path
from typing import Dict

import streamlit as st

from encord_active.app.common.state import get_state
from encord_active.lib.labels.classification import (
    create_ontology_structure,
    update_label_row_with_classification,
)
from encord_active.lib.metrics.execute import get_metrics, load_metric, run_metrics
from encord_active.lib.metrics.metric import EmbeddingType
from encord_active.lib.metrics.utils import get_embedding_type
from encord_active.lib.project.project import Project


class LabelType(str, Enum):
    CLASSIFICATION = "Classification"
    OBJECT = "Object Detection"


LABEL_TYPE_EMBEDDING_MAPPING = {
    LabelType.CLASSIFICATION: EmbeddingType.CLASSIFICATION,
    LabelType.OBJECT: EmbeddingType.OBJECT,
}


SUPPORTED_LABEL_TYPES = [LabelType.CLASSIFICATION]


def label_onboarding_page():
    st.header("Importing labels flow")
    st.info("This is a questionaire that will help us import your labels")
    st.write("")

    label_type = st.selectbox("1. What is the label type?", [v.value for v in LabelType])

    if not label_type:
        return
    elif label_type == LabelType.CLASSIFICATION:
        st.info(
            """
            For classifications, we only support datasets where the data is split accros class directories.
            For example, the classes `cat`, `dog`, and `horse` would be picked up from this file structure:
            ```
            ├── cat/
            │   ├── 0.png
            │   └── ...
            ├── dog/
            │   ├── 0.png
            │   └── ...
            └── horse/
                ├── 0.png
                └── ...
            ```
            """
        )

        project = Project(get_state().project_paths.project_dir).load()
        lr_data_units = [lr["data_units"] for lr in project.label_rows.values()]
        paths = [Path(data_unit["data_link"]) for data_units in lr_data_units for data_unit in data_units.values()]
        image_to_class_map = {path.name: path.parent.stem for path in paths}
        classes = set(image_to_class_map.values())
        class_names_string = ", ".join(f"`{name}`" for name in classes)

        ontology = create_ontology_structure(classes)

        st.subheader(f"We have identified {len(classes)} classes: {class_names_string}")
    else:
        st.warning(f"We currently don't support easy import for {label_type}")
        return

    with st.form("label_import_form"):
        st.write(
            'Press "Import Labels" if you would like to continue with these classes and create the following things:'
        )
        st.expander("Ontology structure").json(ontology.to_dict())
        with st.expander("What metrics do you want to run on your labels?"):
            selected_metrics = render_metric_selection(label_type)

        if st.form_submit_button("Import Labels"):
            with st.spinner():
                project.save_ontology(ontology)

                for label_row in project.label_rows.values():
                    image_class = image_to_class_map[label_row["data_title"]]
                    updated_label_row = update_label_row_with_classification(
                        label_row, ontology.classifications[0], image_class
                    )
                    project.save_label_row(updated_label_row)

                selected_metric_titles = [title for title, should_run in selected_metrics.items() if should_run]
                run_metrics(
                    lambda metric: metric.TITLE in selected_metric_titles,
                    data_dir=get_state().project_paths.project_dir,
                    use_cache_only=True,
                )
            st.experimental_rerun()


def render_metric_selection(label_type: LabelType):
    metrics = map(load_metric, get_metrics())
    selected_metrics: Dict[str, bool] = {}
    for metric in metrics:
        if not metric.ANNOTATION_TYPE:
            continue

        annotation_type = (
            metric.ANNOTATION_TYPE if isinstance(metric.ANNOTATION_TYPE, list) else [metric.ANNOTATION_TYPE]
        )
        embedding_type = get_embedding_type(metric.TITLE, [a for a in annotation_type if a])
        if embedding_type == LABEL_TYPE_EMBEDDING_MAPPING[label_type]:
            selected_metrics[metric.TITLE] = st.checkbox(metric.TITLE, True)

    return selected_metrics
