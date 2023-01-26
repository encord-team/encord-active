import json
from datetime import datetime
from typing import Tuple, cast

import pandas as pd
import streamlit as st
from pandas.api.types import (
    is_categorical_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)

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
        if to_filter_columns:
            df = df.copy()

        for column in to_filter_columns:
            left, right = st.columns((1, 20))
            left.write("↳")
            key = f"filter-select-{column.replace(' ', '_').lower()}"

            if column == "tags":
                tag_filters = right.multiselect(
                    "Choose tags to filter", options=Tags().all(), format_func=lambda x: x.name, key=key
                )
                filtered_rows = [True if set(tag_filters) <= set(x) else False for x in df["tags"]]
                df = df.loc[filtered_rows]

            # Treat columns with < 10 unique values as categorical
            elif is_categorical_dtype(df[column]) or df[column].nunique() < 10:
                if df[column].isnull().sum() != df.shape[0]:
                    user_cat_input = right.multiselect(
                        f"Values for {column}",
                        df[column].dropna().unique().tolist(),
                        default=df[column].dropna().unique().tolist(),
                        key=key,
                    )
                    df = df[df[column].isin(user_cat_input)]
                else:
                    right.markdown(f"For column *{column}*, all values are NaN")
            elif is_numeric_dtype(df[column]):
                _min = float(df[column].min())
                _max = float(df[column].max())
                step = (_max - _min) / 100
                user_num_input = right.slider(
                    f"Values for {column}",
                    min_value=_min,
                    max_value=_max,
                    value=(_min, _max),
                    step=step,
                    key=key,
                )
                df = df[df[column].between(*user_num_input)]
            elif is_datetime64_any_dtype(df[column]):
                first = df[column].min()
                last = df[column].max()
                res = right.date_input(
                    f"Values for {column}", min_value=first, max_value=last, value=(first, last), key=key
                )
                if isinstance(res, tuple) and len(res) == 2:
                    res = cast(Tuple[pd.Timestamp, pd.Timestamp], map(pd.to_datetime, res))  # type: ignore
                    start_date, end_date = res
                    df = df.loc[df[column].between(start_date, end_date)]
            else:
                user_text_input = right.text_input(
                    f"Substring or regex in {column}",
                    key=key,
                )
                if user_text_input:
                    df = df[df[column].astype(str).str.contains(user_text_input)]

    return df


def export_filter():
    get_filtered_row_count, set_filtered_row_count = use_state(0)
    get_clone_button, set_clone_button = use_state(False)

    setup_page()
    message_placeholder = st.empty()

    st.header("Filter & Export")

    filtered_df = filter_dataframe(get_state().merged_metrics.copy())
    filtered_df.reset_index(inplace=True)
    row_count = filtered_df.shape[0]
    st.markdown(f"**Total row:** {row_count}")
    st.dataframe(filtered_df, use_container_width=True)

    action_columns = st.columns((3, 3, 2, 2, 2, 2, 2))
    is_pressed = action_columns[0].button(
        "🌀 Generate COCO file",
        help="Generate COCO file with filtered data",
    )

    with st.spinner(text="Generating COCO file"):
        coco_json = (
            generate_coco_file(filtered_df, get_state().project_paths.project_dir, get_state().project_paths.ontology)
            if is_pressed
            else ""
        )

    action_columns[1].download_button(
        "⬇ Download filtered data",
        json.dumps(coco_json, indent=2),
        file_name=f"encord-active-coco-filtered-{datetime.now().strftime('%Y_%m_%d %H_%M_%S')}.json",
        disabled=not is_pressed,
        help="Ensure you have generated an updated COCO file before downloading",
    )

    action_columns[3].button(
        "🏗 Clone",
        on_click=lambda: set_clone_button(True),
        help="Clone the filtered data into a new Encord dataset and project",
    )
    delete_btn = action_columns[4].button("👀 Review", help="Assign the filtered data for review on the Encord platform")
    edit_btn = action_columns[5].button(
        "🖋 Re-label", help="Assign the filtered data for relabelling on the Encord platform"
    )
    augment_btn = action_columns[6].button("➕ Augment", help="Augment your dataset based on the filered data")

    if any([delete_btn, edit_btn, augment_btn]):
        set_clone_button(False)
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

    prev_row_count = get_filtered_row_count()
    if prev_row_count != row_count:
        set_filtered_row_count(row_count)
        set_clone_button(False)

    if get_clone_button():
        with st.form("new_project_form"):
            st.subheader("Create a new project with the selected items")
            l_column, r_column = st.columns(2)

            dataset_title = l_column.text_input(
                "Dataset title", value=f"Subset: {st.session_state.project_dir.name} ({filtered_df.shape[0]})"
            )
            dataset_description = l_column.text_area("Dataset description")

            project_title = r_column.text_input(
                "Project title", value=f"Sub-project: {st.session_state.project_dir.name} ({filtered_df.shape[0]})"
            )
            project_description = r_column.text_area("Project description")

            if not st.form_submit_button("➕ Create"):
                return

            if dataset_title == "":
                st.error("Dataset title cannot be empty!")
                return
            if project_title == "":
                st.error("Project title cannot be empty!")
                return

            try:
                action_utils = EncordActions(st.session_state.project_dir)
                label = st.empty()
                progress, clear = render_progress_bar()
                label.text("Step 1/2: Uploading data...")
                dataset_creation_result = action_utils.create_dataset(
                    dataset_title, dataset_description, filtered_df, progress
                )
                clear()
                label.text("Step 2/2: Uploading labels...")
                new_project = action_utils.create_project(
                    dataset_creation_result, project_title, project_description, progress
                )
                clear()
                label.info("🎉 New project is created!")

                new_project_link = f"https://app.encord.com/projects/view/{new_project.project_hash}/summary"
                new_dataset_link = f"https://app.encord.com/datasets/view/{dataset_creation_result.hash}"
                st.markdown(f"[Go to new project]({new_project_link})")
                st.markdown(f"[Go to new dataset]({new_dataset_link})")

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
