import json
from pathlib import Path
from typing import List

import pandas as pd

from encord_active.lib.db.connection import DBConnection
from encord_active.lib.db.tags import Tag, TagScope
from encord_active.lib.labels.classification import ClassificationType
from encord_active.lib.metrics.metric import DataType, EmbeddingType
from encord_active.lib.metrics.utils import load_metric_metadata

TABLE_NAME = "merged_metrics"


def build_merged_metrics(metrics_path: Path) -> pd.DataFrame:
    main_df_images = pd.DataFrame(columns=["identifier"])
    main_df_objects = pd.DataFrame(columns=["identifier"])
    main_df_image_quality = pd.DataFrame(columns=["identifier"])

    for index in metrics_path.glob("*.csv"):
        meta_pth = index.with_suffix(".meta.json")
        if not meta_pth.is_file():
            continue

        meta = load_metric_metadata(meta_pth)

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
            main_df_images.rename(columns={"score": f"{meta.title}"}, inplace=True)
        else:
            # Image-level quality index
            if (
                meta.annotation_type[0] == ClassificationType.RADIO
                and meta.data_type == DataType.IMAGE
                and meta.embedding_type == EmbeddingType.CLASSIFICATION
            ):
                main_df_image_quality = metric_scores
                main_df_image_quality.rename(columns={"score": f"{meta.title}"}, inplace=True)
                continue

            # Object-level index
            if main_df_objects.shape[0] == 0:
                columns_to_merge = metric_scores[["identifier", "url", "score"]]
            else:
                columns_to_merge = metric_scores[["identifier", "score"]]
            main_df_objects = pd.merge(main_df_objects, columns_to_merge, how="outer", on="identifier")
            main_df_objects.rename(columns={"score": f"{meta.title}"}, inplace=True)

    main_df = pd.concat([main_df_images, main_df_objects, main_df_image_quality])
    main_df["tags"] = [[] for _ in range(len(main_df))]
    main_df.set_index("identifier", inplace=True)
    return main_df


def marshall_tags(tags: List[Tag]) -> str:
    return json.dumps(tags)


def unmarshall_tags(tags_json: str) -> List[Tag]:
    return [Tag(tag[0], TagScope(tag[1])) for tag in json.loads(tags_json) or []]


def ensure_initialised(fn):
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            merged_metrics = build_merged_metrics(DBConnection.project_file_structure().metrics)
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
    def update_tags(self, id: str, tags: List[Tag]):
        with DBConnection() as conn:
            conn.execute(f"UPDATE {TABLE_NAME} SET tags = '{marshall_tags(tags)}' WHERE IDENTIFIER = '{id}'")

    @ensure_initialised
    def all(self):
        with DBConnection() as conn:
            merged_metrics = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn, index_col="identifier")
            merged_metrics.tags = merged_metrics.tags.apply(unmarshall_tags)
            return merged_metrics

    def replace_all(self, df: pd.DataFrame):
        with DBConnection() as conn:
            copy = df.copy()
            copy.tags = copy.tags.apply(marshall_tags)
            copy.to_sql(name=TABLE_NAME, con=conn, if_exists="replace", index=True, index_label="identifier")

    @ensure_initialised
    def replace_identifiers(self, mappings: dict[str, str]):
        def _replace_identifiers(id: str):
            lr, du, *rest = id.split("_")
            mapped_lr, mapped_du = mappings.get(lr, lr), mappings.get(du, du)
            return "_".join([mapped_lr, mapped_du, *rest])

        with DBConnection() as conn:
            df = pd.read_sql(f"SELECT * FROM {TABLE_NAME}", conn, index_col="identifier")
            df.index = df.index.map(_replace_identifiers)
            df.to_sql(name=TABLE_NAME, con=conn, if_exists="replace", index=True, index_label="identifier")
