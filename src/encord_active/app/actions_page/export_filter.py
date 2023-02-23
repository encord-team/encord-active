import json
from datetime import datetime
from typing import Callable, NamedTuple, Optional, Tuple, cast

import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)
from streamlit.delta_generator import DeltaGenerator

from encord_active.app.app_config import app_config
from encord_active.app.common.state import get_state
from encord_active.app.common.state_hooks import use_state
from encord_active.app.common.utils import set_page_config, setup_page
from encord_active.lib.coco.encoder import generate_coco_file
from encord_active.lib.common.utils import ProjectNotFound
from encord_active.lib.constants import ENCORD_EMAIL, SLACK_URL
from encord_active.lib.db.tags import Tags
from encord_active.lib.encord.actions import (  # create_a_new_dataset,; create_new_project_on_encord_platform,; get_project_user_client,
    EncordActions,
)


def filter_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a UI on top of a dataframe to let apps filter columns

    Args:
        df (pd.DataFrame): Original dataframe

    Returns:
        pd.DataFrame: Filtered dataframe
    """
    with st.container():
        columns_to_filter = df.columns.to_list()
        if "url" in columns_to_filter:
            columns_to_filter.remove("url")
        # Move tags to first option.
        tags_column = columns_to_filter.pop(columns_to_filter.index("tags"))
        columns_to_filter = [tags_column] + columns_to_filter

        to_filter_columns = st.multiselect("Filter dataframe on", columns_to_filter)
        filtered = df.copy()

        for column in to_filter_columns:
            non_applicable = filtered[pd.isna(filtered[column])]

            left, right = st.columns((1, 20))
            left.write("↳")
            key = f"filter-select-{column.replace(' ', '_').lower()}"

            if column == "tags":
                tag_filters = right.multiselect(
                    "Choose tags to filter", options=Tags().all(), format_func=lambda x: x.name, key=key
                )
                filtered_rows = [True if set(tag_filters) <= set(x) else False for x in filtered["tags"]]
                filtered = filtered.loc[filtered_rows]

            # Treat columns with < 10 unique values as categorical
            elif is_categorical_dtype(filtered[column]) or filtered[column].nunique() < 10:
                if filtered[column].isnull().sum() != filtered.shape[0]:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        filtered[column].dropna().unique().tolist(),
                        default=filtered[column].dropna().unique().tolist(),
                        key=key,
                    )
                    filtered = filtered[filtered[column].isin(user_cat_input)]
                else:
                    right.markdown(f"For column *{column}*, all values are NaN")
            elif is_numeric_dtype(filtered[column]):
                _min = float(filtered[column].min())
                _max = float(filtered[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                    key=key,
                )
                filtered = filtered[filtered[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(filtered[column]):
                first = filtered[column].min()
                last = filtered[column].max()
                res = right.date_input(
                    f"Values for {column}", min_value=first, max_value=last, value=(first, last), key=key
                )
                if isinstance(res, tuple) and len(res) == 2:
                    res = cast(Tuple[pd.Timestamp, pd.Timestamp], map(pd.to_datetime, res))  # type: ignore
                    start_date, end_date = res
                    filtered = filtered.loc[filtered[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                    key=key,
                )
                if user_text_input:
                    filtered = filtered[filtered[column].astype(str).str.contains(user_text_input)]

            non_applicable["data_row_id"] = non_applicable.index.str.rsplit("_", n=1, expand=True).droplevel(1)
            filtered_objects_df = non_applicable[non_applicable.data_row_id.isin(filtered.index)].drop(
                ["data_row_id"], axis=1
            )

            filtered = pd.concat([filtered, filtered_objects_df])
    return filtered


class InputItem(NamedTuple):
    title: str
    description: str


class RenderItems(NamedTuple):
    dataset: Optional[InputItem] = None
    project: Optional[InputItem] = None
    ontology: Optional[InputItem] = None


def _get_column(col, item, num_rows) -> InputItem:
    return InputItem(
        col.text_input(f"{item} title", value=f"Subset: {get_state().project_paths.project_dir.name} ({num_rows})"),
        col.text_area(f"{item} description"),
    )


def _get_columns(num_rows: int, items_to_render: list[str]) -> RenderItems:
    form_columns = st.columns(len(items_to_render))
    return RenderItems(*[_get_column(col, item, num_rows) for item, col in zip(items_to_render, form_columns)])


def _get_action_utils():
    try:
        return EncordActions(get_state().project_paths.project_dir, app_config.get_ssh_key())
    except ProjectNotFound as e:
        st.markdown(
            f"""
        ❌ No `project_meta.yaml` file in the project folder.
        Please create `project_meta.yaml` file in **{e.project_dir}** folder with the following content
        and try again:
        ``` yaml
        project_hash: <project_hash>
        ssh_key_path: /path/to/your/encord/ssh_key
        ```
        """
        )
    except Exception as e:
        st.error(str(e))


def create_and_sync_remote_project(action_utils: EncordActions, cols: RenderItems, df: pd.DataFrame):
    if not cols.dataset or not cols.project:
        return
    label = st.empty()
    progress, clear = render_progress_bar()
    label.text("Step 1/2: Uploading data...")
    dataset_creation_result = action_utils.create_dataset(cols.dataset.title, cols.dataset.description, df, progress)
    clear()
    label.text("Step 2/2: Uploading labels...")
    ontology_hash = (
        action_utils.create_ontology(cols.ontology.title, cols.ontology.description).ontology_hash
        if cols.ontology
        else action_utils.original_project.get_project().ontology_hash
    )
    new_project = action_utils.create_project(
        dataset_creation_result, cols.project.title, cols.project.description, ontology_hash, progress
    )
    clear()
    label.empty()
    try:
        action_utils.replace_uids(
            dataset_creation_result.lr_du_mapping, new_project.project_hash, dataset_creation_result.hash
        )
    except Exception as e:
        st.error(str(e))
        return False
    label = st.empty()
    label.info("🎉 New project is created!")

    new_project_link = f"https://app.encord.com/projects/view/{new_project.project_hash}/summary"
    new_dataset_link = f"https://app.encord.com/datasets/view/{dataset_creation_result.hash}"
    st.markdown(f"[Go to new project]({new_project_link})")
    st.markdown(f"[Go to new dataset]({new_dataset_link})")
    return True


def export_filter():
    original_row_count = get_state().merged_metrics.shape[0]
    get_filtered_row_count, set_filtered_row_count = use_state(0)

    get_export_form_button, set_export_form_button = use_state(False)
    get_clone_form_button, set_clone_form_button = use_state(False)

    def close_all_forms(curr_form: Optional[Callable] = None):
        form_controls = [set_export_form_button, set_clone_form_button]
        for form in form_controls:
            if form != curr_form:
                form(False)

    setup_page()
    message_placeholder = st.empty()

    st.header("Filter & Export")
    action_utils = _get_action_utils()
    project_has_remote = action_utils.project_meta.get("has_remote", False) if action_utils else False
    filtered_df = filter_dataframe(get_state().merged_metrics.copy())
    filtered_df.reset_index(inplace=True)
    row_count = filtered_df.shape[0]
    st.markdown(f"**Total row:** {row_count}")
    st.dataframe(filtered_df, use_container_width=True)

    action_columns = st.columns((3, 3, 1, 3, 3, 2, 2, 2))
    file_prefix = get_state().project_paths.project_dir.name

    render_generate_csv(action_columns[0], file_prefix, filtered_df)
    render_generate_coco(action_columns[1], file_prefix, filtered_df)
    render_unimplemented_buttons(
        action_columns[5], action_columns[6], action_columns[7], message_placeholder, close_all_forms
    )

    if get_filtered_row_count() != row_count:
        set_filtered_row_count(row_count)
        close_all_forms()

    if not project_has_remote:
        render_export_button(
            action_columns[3], action_utils, close_all_forms, get_export_form_button, set_export_form_button
        )

    if row_count != original_row_count:
        render_subset_button(
            action_columns[4],
            action_utils,
            filtered_df,
            project_has_remote,
            close_all_forms,
            get_clone_form_button,
            set_clone_form_button,
        )


def generate_create_project_form(header: str, num_rows: int, items_to_render: list[str]) -> Optional[RenderItems]:
    with st.form("new_project_form"):
        st.subheader(header)

        cols = _get_columns(num_rows, items_to_render)
        if not st.form_submit_button("➕ Create"):
            return None
        for item, render_item in zip(cols._fields, cols):
            if render_item and render_item.title == "":
                st.error(f"{item.capitalize()} title cannot be empty!")
                return None
    return cols


def render_subset_button(
    render_col: DeltaGenerator,
    action_utils: EncordActions,
    subset_df: pd.DataFrame,
    project_has_remote: bool,
    close_forms: Callable,
    get_create_button: Callable,
    set_create_button: Callable,
):
    render_col.button(
        "🏗 Create Subset",
        on_click=lambda: (close_forms(curr_form=set_create_button), set_create_button(True)),
        disabled=not action_utils,
        help="Subset the filtered data into a new Encord dataset and project",
    )

    if get_create_button():
        items_to_render = ["Project"] if not project_has_remote else ["Dataset", "Project"]
        cols = generate_create_project_form(
            "Create a subset with the selected items", subset_df.shape[0], items_to_render
        )
        if not cols or not cols.project:
            return

        with st.spinner("Creating Project Subset..."):
            try:
                created_dir = action_utils.create_local_subset(
                    project_title=cols.project.title,
                    project_description=cols.project.description,
                    filtered_df=subset_df,
                )
            except Exception as e:
                st.error(str(e))
            if project_has_remote and cols.dataset:
                action_utils.clone_remote_copy(
                    created_dir,
                    cols.project.title,
                    cols.project.description,
                    cols.dataset.title,
                    cols.dataset.description,
                    subset_df,
                )
        label = st.empty()
        label.info("🎉 Project subset has been created!")
        st.markdown(created_dir)


def render_export_button(
    render_col: DeltaGenerator,
    action_utils: EncordActions,
    close_forms: Callable,
    get_export_button: Callable,
    set_export_button: Callable,
):
    export_button = render_col.button(
        "🏗 Export to Encord",
        on_click=lambda: (close_forms(curr_form=set_export_button), set_export_button(True)),
        disabled=not action_utils,
        help="Export to an Encord dataset and project",
    )
    df = get_state().merged_metrics

    if get_export_button():
        items_to_render = ["Dataset", "Project", "Ontology"]
        cols = generate_create_project_form(
            "Create a new project with the current dataset", df.shape[0], items_to_render
        )
        if not cols:
            return

        success = create_and_sync_remote_project(action_utils, cols, df)


def render_unimplemented_buttons(
    delete_button_column: DeltaGenerator,
    edit_button_column: DeltaGenerator,
    augment_button_column: DeltaGenerator,
    message_placeholder: DeltaGenerator,
    close_all_forms: Callable,
):
    delete_btn = delete_button_column.button(
        "👀 Review", help="Assign the filtered data for review on the Encord platform"
    )
    edit_btn = edit_button_column.button(
        "🖋 Re-label", help="Assign the filtered data for relabelling on the Encord platform"
    )
    augment_btn = augment_button_column.button("➕ Augment", help="Augment your dataset based on the filtered data")
    if any([delete_btn, edit_btn, augment_btn]):
        close_all_forms()
        message_placeholder.markdown(
            f"""
<div class="encord-active-info-box">
    For more information and help on using this feature reach out to us on the
    <a href="{SLACK_URL}" target="_blank"><i class="fa-brands fa-slack"></i> Encord Active Slack
community</a>
    or shoot us an
    <a href="mailto:{ENCORD_EMAIL}" target="_blank"><i class="fa fa-envelope"></i> email</a>🙏
</div>
""",
            unsafe_allow_html=True,
        )


def render_generate_coco(render_column: DeltaGenerator, file_prefix: str, filtered_df: pd.DataFrame):
    coco_placeholder = render_column.empty()
    generate_coco = coco_placeholder.button(
        "🌀 Generate COCO",
        help="Generate COCO file with filtered data to enable COCO download.",
    )
    if generate_coco:
        with st.spinner(text="Generating COCO file"):
            coco_json = (
                generate_coco_file(
                    filtered_df, get_state().project_paths.project_dir, get_state().project_paths.ontology
                )
                if generate_coco
                else ""
            )
            coco_placeholder.empty()

        coco_placeholder.download_button(
            "⬇ Download COCO",
            json.dumps(coco_json, indent=2),
            file_name=f"{file_prefix}-coco-filtered-{datetime.now().strftime('%Y_%m_%d %H_%M_%S')}.json",
            help="Ensure you have generated an updated COCO file before downloading",
        )


def render_generate_csv(render_column: DeltaGenerator, file_prefix: str, filtered_df: pd.DataFrame):
    csv_placeholder = render_column.empty()
    generate_csv = csv_placeholder.button(
        "🌀 Generate CSV",
        help="Generate CSV file with filtered data to enable CSV download.",
    )
    if generate_csv:
        with st.spinner(text="Generating CSV file"):
            csv_content = filtered_df.to_csv().encode("utf-8") if generate_csv else ""
            csv_placeholder.empty()

        csv_placeholder.download_button(
            "⬇ Download CSV",
            data=csv_content,
            file_name=f"{file_prefix}-filtered-{datetime.now().strftime('%Y_%m_%d %H_%M_%S')}.csv",
            help="Ensure you have generated an updated COCO file before downloading",
        )


def render_progress_bar():
    progress_bar = st.empty()

    def clear():
        progress_bar.empty()

    def progress_callback(value: float):
        progress_bar.progress(value)

    return progress_callback, clear


if __name__ == "__main__":
    set_page_config()
    export_filter()
