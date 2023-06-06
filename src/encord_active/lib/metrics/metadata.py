import json
from pathlib import Path

from encord_active.lib.metrics.io import fill_metrics_meta_with_builtin_metrics
from encord_active.lib.project import ProjectFileStructure


def fetch_metrics_meta(project_file_structure: ProjectFileStructure):
    meta_file = project_file_structure.metrics_meta
    builtin_metrics = fill_metrics_meta_with_builtin_metrics()

    if not meta_file.is_file():
        metrics_meta = builtin_metrics
    else:
        metrics_meta = json.loads(meta_file.read_text(encoding="utf-8"))
        for metric_title in builtin_metrics:
            if metric_title in metrics_meta and not Path(metrics_meta[metric_title]["location"]).exists():
                metrics_meta[metric_title] = builtin_metrics[metric_title]

    update_metrics_meta(project_file_structure, metrics_meta)  # Update refreshed metric locations
    return metrics_meta


def update_metrics_meta(project_file_structure: ProjectFileStructure, updated_meta: dict):
    meta_file_path = project_file_structure.metrics_meta
    meta_file_path.parent.mkdir(exist_ok=True)
    meta_file_path.write_text(json.dumps(updated_meta, indent=2), encoding="utf-8")
