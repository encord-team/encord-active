import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Callable, List, NamedTuple, Optional, Tuple, cast

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
from encord_active.app.common.state_hooks import UseState
from encord_active.app.common.utils import human_format, set_page_config
from encord_active.lib.coco.encoder import generate_coco_file
from encord_active.lib.constants import DISCORD_URL, ENCORD_EMAIL
from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.helpers.tags import all_tags
from encord_active.lib.db.merged_metrics import MergedMetrics
from encord_active.lib.db.tags import TagScope
from encord_active.lib.encord.actions import DatasetUniquenessError, EncordActions
from encord_active.lib.encord.utils import get_encord_project
from encord_active.lib.metrics.utils import load_metric_dataframe
from encord_active.lib.project import ProjectFileStructure
from encord_active.lib.project.metadata import ProjectNotFound, fetch_project_meta


@dataclass
class ProjectCreationResult:
    project_hash: str
    dataset_hash: str


class UpdateItemType(Enum):
    LABEL = auto()
    MARKDOWN = auto()


@dataclass
class UpdateItem:
    type: UpdateItemType
    text: str


class CurrentForm(str, Enum):
    NONE = "none"
    EXPORT = "export"
    CLONE = "clone"
    RELABEL = "relabel"


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
        # Move some columns to the begining
        columns_to_filter.sort(key=lambda column: column in ["object_class", "tags", "annotator"], reverse=True)

        to_filter_columns = st.multiselect(
            "Filter by", columns_to_filter, format_func=lambda name: name.replace("_", " ").title()
        )

        if not to_filter_columns:
            return df

        filtered = df.copy()
        filtered["data_row_id"] = filtered.index.str.split("_", n=3).str[0:3].str.join("_")
        for column in to_filter_columns:
            non_applicable = filtered[pd.isna(filtered[column])]

            left, right = st.columns((1, 20))
            left.write("↳")
            key = f"filter-select-{column.replace(' ', '_').lower()}"

            if column == "tags":
                tag_filters = right.multiselect(
                    "Choose tags to filter",
                    options=all_tags(get_state().project_paths),
                    format_func=lambda x: x.name,
                    key=key,
                )
                for tag in tag_filters:
                    filtered_rows = [tag in x for x in filtered["tags"]]
                    filtered_items = filtered.loc[filtered_rows]
                    if tag.scope == TagScope.LABEL:
                        non_applicable = filtered[filtered.index.isin(filtered_items["data_row_id"])]
                        filtered = filtered[filtered.index.isin(filtered_items.index)]
                    else:
                        filtered = filtered[filtered.data_row_id.isin(filtered_items["data_row_id"])]

            # Treat columns with < 10 unique values as categorical
            elif (
                is_categorical_dtype(filtered[column])
                or filtered[column].nunique() < 10
                or column in ["object_class", "annotator"]
            ):
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
            if not non_applicable.empty:
                filtered_objects_df = non_applicable[non_applicable.data_row_id.isin(filtered["data_row_id"])]
                filtered = pd.concat([filtered, filtered_objects_df])

    return filtered.drop("data_row_id", axis=1)


class InputItem(NamedTuple):
    title: str
    description: str


class RenderItems(NamedTuple):
    project: Optional[InputItem] = None
    dataset: Optional[InputItem] = None
    ontology: Optional[InputItem] = None


def _get_column(col: DeltaGenerator, item: str, num_rows: int, subset: bool, project_name: str) -> InputItem:
    return InputItem(
        col.text_input(
            f"{item} title",
            value=f"{'Subset: ' if subset else ''}{project_name} ({num_rows})",
        ),
        col.text_area(f"{item} description"),
    )


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


def create_and_sync_remote_project(
    action_utils: EncordActions, cols: RenderItems, df: pd.DataFrame
) -> Optional[ProjectCreationResult]:
    if not cols.dataset or not cols.project:
        return None
    label = st.empty()
    progress, clear = render_progress_bar()
    label.text("Step 1/2: Uploading data...")
    try:
        dataset_creation_result = action_utils.create_dataset(
            cols.dataset.title, cols.dataset.description, df, progress
        )
    except DatasetUniquenessError as e:
        clear()
        label.empty()
        st.error(str(e))
        return None
    clear()
    label.text("Step 2/2: Uploading labels...")
    ontology_hash = (
        action_utils.create_ontology(cols.ontology.title, cols.ontology.description).ontology_hash
        if cols.ontology
        else action_utils.original_project.get_project().ontology_hash
    )

    try:
        new_project = action_utils.create_project(
            dataset_creation_result, cols.project.title, cols.project.description, ontology_hash, progress
        )
    except Exception as e:
        st.error(str(e))
        return None
    clear()
    label.empty()
    return ProjectCreationResult(project_hash=new_project.project_hash, dataset_hash=dataset_creation_result.hash)


def show_update_stats(filtered_df: pd.DataFrame):
    frames, labels = st.columns(2)
    project_path = get_state().project_paths.project_dir

    def get_key(counter_name: str):
        return f"{counter_name}_{project_path}"

    state_frame_count = UseState[Optional[int]](None, key=get_key("FILTER_EXPORT_FRAME_COUNT"))
    frame_count: int = len(filtered_df.index.map(lambda x: tuple(x.split("_")[1:3])).unique())
    with frames:
        st.metric(
            "Selected Frames",
            value=human_format(frame_count),
            delta=state_frame_count.value and f"{human_format(frame_count - state_frame_count.value)} frames",
        )
        state_frame_count.set(frame_count)

    state_label_count = UseState[Optional[int]](None, key=get_key("FILTER_EXPORT_LABEL_COUNT"))
    label_count: int = filtered_df[filtered_df.index.map(lambda x: len(x.split("_")) > 3)].shape[0]
    with labels:
        st.metric(
            "Selected Labels",
            value=human_format(label_count),
            delta=state_label_count.value and f"{human_format(label_count - state_label_count.value)} labels",
        )
        state_label_count.set(label_count)


def render_filter():
    filter_col, _, stats_col = st.columns([8, 1, 2])
    with filter_col:
        with DBConnection(get_state().project_paths) as conn:
            filtered_merged_metrics = filter_dataframe(MergedMetrics(conn).all())

    with stats_col:
        show_update_stats(filtered_merged_metrics)

    get_state().filtering_state.merged_metrics = filtered_merged_metrics


def actions():
    original_row_count = get_state().merged_metrics.shape[0]
    filtered_row_count = UseState(0)

    current_form = UseState(CurrentForm.NONE)

    updates = UseState[List[UpdateItem]]([])

    message_placeholder = st.empty()

    action_utils = _get_action_utils()
    project_has_remote = bool(action_utils.project_meta.get("has_remote", False)) if action_utils else False
    project_name = (
        action_utils.project_meta.get("project_title", get_state().project_paths.project_dir.name)
        if action_utils
        else get_state().project_paths.project_dir.name
    )

    filtered_df = get_state().filtering_state.merged_metrics.reset_index()
    if filtered_df is None:
        filtered_df = get_state().merged_metrics

    row_count = filtered_df.shape[0]

    (
        generate_csv_col,
        generate_coco_col,
        export_button_col,
        subset_button_col,
        delete_button_col,
        relabel_button_col,
        augment_button_col,
    ) = st.columns((3, 3, 3, 3, 2, 2, 2))
    file_prefix = get_state().project_paths.project_dir.name

    render_generate_csv(generate_csv_col, file_prefix, filtered_df)
    render_generate_coco(generate_coco_col, file_prefix, filtered_df)
    render_relabel_button(relabel_button_col, get_state().project_paths, filtered_df, current_form)
    render_unimplemented_buttons(delete_button_col, augment_button_col, message_placeholder, current_form.set)

    if filtered_row_count.value != row_count:
        filtered_row_count.set(row_count)
        current_form.set(CurrentForm.NONE)

    if not project_has_remote:
        render_export_button(
            export_button_col,
            action_utils,
            current_form,
            project_name,
            updates.set,
            is_filtered=(row_count != original_row_count),
        )

    if row_count != original_row_count:
        render_subset_button(
            subset_button_col,
            action_utils,
            filtered_df,
            project_has_remote,
            current_form,
            project_name,
            updates.set,
        )

    for update in updates.value:
        if update.type == UpdateItemType.LABEL:
            label = st.empty()
            label.info(update.text)
        elif update.type == UpdateItemType.MARKDOWN:
            st.markdown(update.text)
    updates.set([])


def generate_create_project_form(
    header: str, num_rows: int, dataset: bool, ontology: bool, subset: bool, project_name: str
) -> Optional[RenderItems]:
    with st.form("new_project_form"):
        st.subheader(header)
        items_to_render = ["Project"]
        if dataset:
            items_to_render.append("Dataset")
        if ontology:
            items_to_render.append("Ontology")

        form_columns = st.columns(len(items_to_render))
        cols = RenderItems(
            *[
                _get_column(col, item, num_rows, subset, project_name=project_name)
                for item, col in zip(items_to_render, form_columns)
            ]
        )

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
    current_form: UseState[CurrentForm],
    project_name: str,
    set_updates: Callable[[List[UpdateItem]], None],
):
    render_col.button(
        "🏗 Create Subset",
        on_click=lambda: current_form.set(CurrentForm.CLONE),  # type: ignore
        disabled=not action_utils,
        help="Subset the filtered data into a new Encord dataset and project",
    )

    if current_form.value == CurrentForm.CLONE:
        cols = generate_create_project_form(
            "Create a subset with the selected items",
            subset_df.shape[0],
            dataset=project_has_remote,
            ontology=False,
            subset=True,
            project_name=project_name,
        )
        if not cols or not cols.project:
            return

        with st.spinner("Creating Project Subset..."):
            try:
                created_dir = action_utils.create_subset(
                    filtered_df=subset_df,
                    remote_copy=bool(project_has_remote),
                    project_title=cols.project.title,
                    project_description=cols.project.description,
                    dataset_title=cols.dataset.title if cols.dataset else None,
                    dataset_description=cols.dataset.description if cols.dataset else None,
                )
            except Exception as e:
                st.error(str(e))
                return

        set_updates(
            [
                UpdateItem(
                    type=UpdateItemType.LABEL,
                    text=f"🎉 Project subset has been created! You can view it at {created_dir.as_posix()}",
                )
            ]
        )
        get_state().refresh_projects()


def render_export_button(
    render_col: DeltaGenerator,
    action_utils: EncordActions,
    current_form: UseState[CurrentForm],
    project_name: str,
    set_updates: Callable[[List[UpdateItem]], None],
    is_filtered: bool,
):
    render_col.button(
        "🏗 Export to Encord",
        on_click=lambda: current_form.set(CurrentForm.EXPORT),  # type: ignore
        disabled=not action_utils or is_filtered,
        help="Export to an Encord dataset and project"
        if not is_filtered
        else "Export is allowed only for entire datasets, create a subset first or remove all filters",
    )
    if current_form.value == CurrentForm.EXPORT:
        df = get_state().merged_metrics
        cols = generate_create_project_form(
            "Create a new project with the current dataset",
            df.shape[0],
            dataset=True,
            ontology=True,
            subset=False,
            project_name=project_name,
        )

        if not cols:
            return
        project_creation_result = create_and_sync_remote_project(action_utils, cols, df)
        if project_creation_result:
            # Clean summaries' cache to avoid hit old data hashes
            get_state().metrics_data_summary = None
            get_state().metrics_label_summary = None
            load_metric_dataframe.cache_clear()
            get_state().project_paths.cache_clear()

            new_project_link = f"https://app.encord.com/projects/view/{project_creation_result.project_hash}/summary"
            new_dataset_link = f"https://app.encord.com/datasets/view/{project_creation_result.dataset_hash}"
            update_items = [
                UpdateItem(type=UpdateItemType.LABEL, text="🎉 Project exported to Encord Annotate!"),
                UpdateItem(type=UpdateItemType.MARKDOWN, text=f"[Go to new project]({new_project_link})"),
                UpdateItem(type=UpdateItemType.MARKDOWN, text=f"[Go to new dataset]({new_dataset_link})"),
            ]
            set_updates(update_items)


def render_unimplemented_buttons(
    delete_button_column: DeltaGenerator,
    augment_button_column: DeltaGenerator,
    message_placeholder: DeltaGenerator,
    set_current_form: Callable,
):
    delete_btn = delete_button_column.button(
        "👀 Review", help="Assign the filtered data for review on the Encord platform"
    )
    augment_btn = augment_button_column.button("➕ Augment", help="Augment your dataset based on the filtered data")
    if any([delete_btn, augment_btn]):
        set_current_form(CurrentForm.NONE)
        message_placeholder.markdown(
            f"""
<div class="encord-active-info-box">
    For more information and help on using this feature reach out to us on the
    <a href="{DISCORD_URL}" target="_blank"><i class="fa-brands fa-discord"></i> Encord Active Discord
community</a>
    or shoot us an
    <a href="mailto:{ENCORD_EMAIL}" target="_blank"><i class="fa fa-envelope"></i> email</a>🙏
</div>
""",
            unsafe_allow_html=True,
        )


def render_generate_coco(
    render_column: DeltaGenerator,
    file_prefix: str,
    filtered_df: pd.DataFrame,
):
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


def render_generate_csv(
    render_column: DeltaGenerator,
    file_prefix: str,
    filtered_df: pd.DataFrame,
):
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


def render_relabel_button(
    render_column: DeltaGenerator,
    pfs: ProjectFileStructure,
    filtered_df: pd.DataFrame,
    current_form: UseState[CurrentForm],
):
    project_meta = fetch_project_meta(pfs.project_dir)
    label_row_meta: dict[str, dict] = json.loads(pfs.label_row_meta.read_text(encoding="utf-8"))

    # Check the conditions to be able to send the filtered data to labeling on Encord (remote project)
    is_disabled = False
    extra_help_text = ""
    if not project_meta.get("has_remote", False):
        # Shouldn't work with local projects
        is_disabled = True
        extra_help_text = (
            " Relabeling is only supported on workflow projects and this project is local."
            " Please, first export the data to a workflow project in the Encord platform."
        )

    elif len(label_row_meta) == 0 or next(iter(label_row_meta.values())).get("workflow_graph_node") is None:
        # Relabeling is not supported on non-workflow projects
        is_disabled = True
        extra_help_text = (
            " Relabeling is only supported on workflow projects."
            " Please, first upgrade your project so it uses workflows."
        )

    relabel_placeholder = render_column.empty()
    relabel_placeholder.button(
        "🖋 Re-label",
        help="Assign the filtered data for relabeling in the Encord platform." + extra_help_text,
        disabled=is_disabled,
        on_click=lambda: current_form.set(CurrentForm.RELABEL),
    )

    if current_form.value != CurrentForm.RELABEL:
        return

    # Get the label rows to submit for labeling
    all_identifiers = filtered_df["identifier"].to_list()
    unique_lr_hashes = list(set(str(_id).split("_", maxsplit=1)[0] for _id in all_identifiers))

    form_placeholder = st.empty()  # Show the confirmation form
    with form_placeholder.form("relabel_confirmation_form"):
        label = st.empty()
        label.text(
            # Next line should stay commented while 'all_identifiers' has a wrong amount of items
            # f"You have selected {len(all_identifiers)} items to submit for labeling in Encord Annotate.\n"
            f"A total of {len(unique_lr_hashes)} tasks are going to be assigned to the first labeling stage.\n"
            f"Are you sure you want to proceed?"
        )

        if not st.form_submit_button("Confirm"):
            return None
    form_placeholder.empty()  # Disappear the form after clicking the confirmation button

    # Start the submission of the selected label rows for labeling
    label = st.empty()
    label.text("Sending the selected data to labeling.")
    update_progress_bar, clear_progress_bar = render_progress_bar()
    update_progress_bar(0)  # Show progress bar

    encord_project = get_encord_project(project_meta["ssh_key_path"], project_meta["project_hash"])
    for index, label_row in enumerate(encord_project.list_label_rows_v2(label_hashes=unique_lr_hashes), start=1):
        label_row.workflow_reopen()
        update_progress_bar(index / len(unique_lr_hashes))

    clear_progress_bar()
    label.empty()
    label.text(
        f"Data successfully submitted for labeling in Encord Annotate.\n"
        f"A total of {len(unique_lr_hashes)} tasks have been assigned to the first labeling stage."
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
    actions()
