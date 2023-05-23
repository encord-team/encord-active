import json
from math import isinf

from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.metrics.io import fill_metrics_meta_with_builtin_metrics
from encord_active.lib.project import ProjectFileStructure


def fetch_metrics_meta(project_file_structure: ProjectFileStructure):
    with PrismaConnection(project_file_structure) as conn:
        rows = conn.metricmetadata.find_many(
            include={
                "annotation_type": True,
            }
        )
        def convert_row(row):
            return {
                "title": row.title,
                "short_description": row.short_description,
                "metric_type": row.metric_type,
                "data_type": row.data_type,
                "annotation_type": [
                    r["annotation_type"]
                    for r in row.annotation_type
                ],
                "embedding_type": row.embedding_type,
                "doc_url": row.doc_url,
                "stats": {
                    "min_value": row.stat_min_value,
                    "max_value": row.stat_max_value,
                    "mean_value": row.stat_mean_value,
                    "num_rows": row.stat_num_rows,
                }
            }
    result = [convert_row(row) for row in rows]
    if len(result) == 0:
        metrics = fill_metrics_meta_with_builtin_metrics()
        update_metrics_meta(project_file_structure, metrics)

    return fetch_metrics_meta(project_file_structure)


def update_metrics_meta(project_file_structure: ProjectFileStructure, updated_meta: dict):
    for data in updated_meta.values():
        title = data["title"]
        stats = data.pop("stats")
        for k, v in dict(stats).items():
            data[f"stat_{k}"] = float(v) if not isinf(float(v)) else 0.0
        data["stat_num_rows"] = int(data["stat_num_rows"])
        data["annotation_type"] = [] if data.get("annotation_type", None) is None else \
            [{"annotation_type": value} for value in data["annotation_type"]]
        data["annotation_type"] = {
            "create": data["annotation_type"]
        }
        with PrismaConnection(project_file_structure) as conn:
            conn.metricmetadata.upsert(
                where={
                    "title": title,
                },
                data={
                    "create": data,
                    "update": {
                        k: v
                        for k, v in data.items()
                        if k != "annotation_type"
                    }
                },
                include={
                    "annotation_type": True,
                }
            )
