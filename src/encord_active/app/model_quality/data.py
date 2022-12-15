import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, cast

import pandas as pd
import streamlit as st
from natsort import natsorted

from encord_active.app.common.utils import load_json


def check_model_prediction_availability():
    predictions_path = st.session_state.predictions_dir / "predictions.csv"
    return predictions_path.is_file()


def merge_objects_and_scores(
    object_df: pd.DataFrame, metric_pth: Optional[Path] = None, ignore_object_scores=True
) -> Tuple[pd.DataFrame, List[str]]:
    metric_names: List[str] = []
    object_df["identifier_no_oh"] = object_df["identifier"].str.replace(r"^(\S{73}_\d+)(.*)", r"\1", regex=True)

    if metric_pth is not None:
        # Import prediction scores
        for metric in metric_pth.iterdir():
            if not metric.suffix == ".csv":
                continue

            meta_pth = metric.with_suffix(".meta.json")
            if not meta_pth.is_file():
                continue

            with meta_pth.open("r", encoding="utf-8") as f:
                meta = json.load(f)

            metric_scores = pd.read_csv(metric, index_col="identifier")
            # Ignore empty data frames.
            if metric_scores.shape[0] == 0:
                continue

            title = f"{meta['title']} (P)"
            metric_names.append(title)

            has_object_level_keys = len(metric_scores.index[0].split("_")) > 3
            metric_column = "identifier" if has_object_level_keys else "identifier_no_oh"
            # Join data and rename column to metric name.
            object_df = object_df.join(metric_scores["score"], on=[metric_column])
            object_df[title] = object_df["score"]
            object_df.drop("score", axis=1, inplace=True)

    # Import frame level scores
    for metric_file in st.session_state.metric_dir.iterdir():
        if metric_file.is_dir() or metric_file.suffix != ".csv":
            continue

        # Read first row to see if metric has frame level scores
        with metric_file.open("r", encoding="utf-8") as f:
            f.readline()  # header
            key, *_ = f.readline().split(",")

        if not key:  # Empty metric
            continue

        label_hash, du_hash, frame, *rest = key.split("_")
        type_indicator = "F"  # Frame level
        join_column = "identifier_no_oh"
        if rest and ignore_object_scores:
            # There are object hashes included in the key, so ignore.
            continue
        elif rest:
            type_indicator = "O"
            join_column = "identifier"

        meta_pth = metric_file.with_suffix(".meta.json")
        if not meta_pth.is_file():
            continue

        with meta_pth.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        metric_scores = pd.read_csv(metric_file, index_col="identifier")

        title = f"{meta['title']} ({type_indicator})"
        metric_names.append(title)

        # Join data and rename column to metric name.
        object_df = object_df.join(metric_scores["score"], on=join_column)
        object_df[title] = object_df["score"]
        object_df.drop("score", axis=1, inplace=True)
    object_df.drop("identifier_no_oh", axis=1, inplace=True)
    metric_names = cast(List[str], natsorted(metric_names, key=lambda x: x[-3:] + x[:-3]))
    return object_df, metric_names


@st.cache(allow_output_mutation=True)
def get_model_predictions() -> Optional[Tuple[pd.DataFrame, List[str]]]:
    """
    Loads predictions and their associated metric scores.
    :param predictions_path:
    :return:
        - predictions: The predictions with their metric scores
        - column names: Unit test column names.
    """
    predictions_path = st.session_state.predictions_dir / "predictions.csv"
    if not predictions_path.is_file():
        st.error(f"Labels file `{st.session_state.predictions_dir / 'predictions.csv'} is missing.")
        return None

    predictions_df = pd.read_csv(predictions_path)

    # Extract label_hash, du_hash, frame
    identifiers = predictions_df["identifier"].str.split("_", expand=True)
    identifiers.columns = ["label_hash", "du_hash", "frame", "object_hash"][: len(identifiers.columns)]
    identifiers["frame"] = pd.to_numeric(identifiers["frame"])
    predictions_df = pd.concat([predictions_df, identifiers], axis=1)

    # Load predictions scores (metrics)
    pred_idx_pth = st.session_state.predictions_dir / "metrics"
    if not pred_idx_pth.exists():
        return predictions_df, []

    predictions_df, metric_names = merge_objects_and_scores(predictions_df, metric_pth=pred_idx_pth)

    return predictions_df, metric_names


@st.cache(allow_output_mutation=True)
def get_labels() -> Optional[Tuple[pd.DataFrame, List[str]]]:
    labels_path = st.session_state.predictions_dir / "labels.csv"
    if not labels_path.is_file():
        st.error(f"Labels file `{st.session_state.predictions_dir / 'labels.csv'} is missing")
        return None

    labels_df = pd.read_csv(labels_path)

    # Extract label_hash, du_hash, frame
    identifiers = labels_df["identifier"].str.split("_", expand=True)
    identifiers = identifiers.iloc[:, :3]
    identifiers.columns = ["label_hash", "du_hash", "frame"]
    identifiers["frame"] = pd.to_numeric(identifiers["frame"])

    labels_df = pd.concat([labels_df, identifiers], axis=1)
    labels_df, label_metric_names = merge_objects_and_scores(labels_df, ignore_object_scores=False)
    return labels_df, label_metric_names


@st.cache(allow_output_mutation=True)
def get_gt_matched() -> Optional[dict]:
    gt_path = st.session_state.predictions_dir / "ground_truths_matched.json"
    return load_json(gt_path)


@st.cache()
def get_class_idx() -> Optional[dict]:
    class_idx_pth = st.session_state.predictions_dir / "class_idx.json"
    return load_json(class_idx_pth)


@st.cache()
def get_metadata_files() -> Dict[str, dict]:
    out: Dict[str, dict] = {}
    # Look in data metrics
    data_metrics = out.setdefault("data", {})
    if st.session_state.metric_dir.is_dir():
        for f in st.session_state.metric_dir.iterdir():
            if not f.name.endswith(".meta.json"):
                continue

            meta = load_json(f)
            if meta is None:
                continue

            data_metrics[meta["title"]] = meta

    prediction_metrics = out.setdefault("prediction", {})
    if (st.session_state.predictions_dir / "metrics").is_dir():
        for f in st.session_state.metric_dir.iterdir():
            if not f.name.endswith(".meta.json"):
                continue

            meta = load_json(f)
            if meta is None:
                continue

            prediction_metrics[meta["title"]] = meta

    return out


def filter_labels_for_frames_wo_predictions():
    """
    Note: data_root is not used in the code, but utilized by `st` to determine what
    to cache, so please don't remove.
    """
    _predictions = st.session_state.model_predictions
    pred_keys = _predictions["img_id"].unique()
    _labels = st.session_state.sorted_labels
    return _labels[_labels["img_id"].isin(pred_keys)]


def prediction_and_label_filtering(labels, metrics, model_pred, precisions):
    # Filtering based on selection
    # In the following a "_" prefix means the the data has been filtered according to selected classes.
    # Predictions
    class_idx = st.session_state.selected_class_idx
    row_selection = model_pred["class_id"].isin(set(map(int, class_idx.keys())))
    _model_pred = model_pred[row_selection].copy()
    # Labels
    row_selection = labels["class_id"].isin(set(map(int, class_idx.keys())))
    _labels = labels[row_selection]

    chosen_name_set = set(map(lambda x: x["name"], class_idx.values())).union({"Mean"})
    _metrics = metrics[metrics["class_name"].isin(chosen_name_set)]
    _precisions = precisions[precisions["class_name"].isin(chosen_name_set)]
    name_map = {int(k): v["name"] for k, v in class_idx.items()}
    _model_pred["class_name"] = _model_pred["class_id"].map(name_map)
    _labels["class_name"] = _labels["class_id"].map(name_map)
    return _labels, _metrics, _model_pred, _precisions
