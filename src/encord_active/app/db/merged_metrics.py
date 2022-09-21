import json
from typing import List

import pandas as pd
import streamlit as st
from encord.project_ontology.classification_type import ClassificationType

from encord_active.app.common.state import MERGED_DATAFRAME
from encord_active.app.db.connection import DBConnection

TABLE_NAME = "merged_metrics"


def build_merged_metrics() -> pd.DataFrame:
    main_df_images = pd.DataFrame(columns=["identifier"])
    main_df_objects = pd.DataFrame(columns=["identifier"])
    main_df_image_quality = pd.DataFrame()

    for index in st.session_state.metric_dir.glob("*.csv"):
        meta_pth = index.with_suffix(".meta.json")
        if not meta_pth.is_file():
            continue

        with meta_pth.open("r", encoding="utf-8") as f:
            meta = json.load(f)

        metric_scores = pd.read_csv(index)
        if metric_scores.shape[0] == 0:
            continue

        if len(metric_scores["identifier"][0].split("_")) == 3:
            # Image-level index
            if main_df_images.shape[0] == 0:
                columns_to_merge = metric_scores[["identifier", "url", "score"]]
            else:
                columns_to_merge = metric_scores[["identifier", "score"]]
            main_df_images = pd.merge(main_df_images, columns_to_merge, how="outer", on="identifier")
            main_df_images.rename(columns={"score": f"{meta['title']}"}, inplace=True)
        else:
            # Image-level quality index
            if (
                meta["annotation_type"][0] == ClassificationType.RADIO.value
                and meta["data_type"] == "image"
                and meta["embedding_type"] == "classification"
            ):
                main_df_image_quality = metric_scores
                main_df_image_quality.rename(columns={"score": f"{meta['title']}"}, inplace=True)
                continue

            # Object-level index
            if main_df_objects.shape[0] == 0:
                columns_to_merge = metric_scores[["identifier", "url", "score"]]
            else:
                columns_to_merge = metric_scores[["identifier", "score"]]
            main_df_objects = pd.merge(main_df_objects, columns_to_merge, how="outer", on="identifier")
            main_df_objects.rename(columns={"score": f"{meta['title']}"}, inplace=True)

    main_df = pd.concat([main_df_images, main_df_objects, main_df_image_quality])
    main_df["tags"] = None
    main_df.set_index("identifier", inplace=True)
    return main_df


def marshall_tags(df: pd.DataFrame):
    copy = df.copy()
    copy.tags = copy.tags.apply(lambda tags: ",".join(tags or []))
    return copy


def unmarshall_tags(df: pd.DataFrame):
    copy = df.copy()
    copy.tags = copy.tags.apply(lambda tags: [] if not tags else tags.split(","))
    return copy


def ensure_initialised(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except:
            merged_metrics = st.session_state.get(MERGED_DATAFRAME)
            if merged_metrics is None:
                merged_metrics = build_merged_metrics()
            MergedMetrics().replace_all(merged_metrics)
            return fn(*args, **kwargs)

    return wrapper


class MergedMetrics(object):
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(MergedMetrics, cls).__new__(cls)
        return cls.instance

    @ensure_initialised
    def get_row(self, id: str):
        with DBConnection() as conn:
            return pd.read_sql(f"SELECT * FROM {TABLE_NAME} where IDENTIFIER = '{id}'", conn)

    @ensure_initialised
    def update_tags(self, id: str, tags: List[str]):
        with DBConnection() as conn:
            conn.execute(f"UPDATE {TABLE_NAME} SET tags = '{','.join(tags)}' WHERE IDENTIFIER = '{id}'")

    @ensure_initialised
    def all(self):
        with DBConnection() as conn:
            merged_metrics = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn, index_col="identifier")
            return unmarshall_tags(merged_metrics)

    def replace_all(self, df: pd.DataFrame):
        with DBConnection() as conn:
            marshall_tags(df).to_sql(
                name=TABLE_NAME, con=conn, if_exists="replace", index=True, index_label="identifier"
            )
