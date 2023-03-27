import json

from encord_active.lib.metrics.io import fill_metrics_meta_with_builtin_metrics
from encord_active.lib.project import ProjectFileStructure


def fetch_metrics_meta(project_file_structure: ProjectFileStructure):
    meta_file = project_file_structure.metrics_meta
    if not meta_file.is_file():
        metrics = fill_metrics_meta_with_builtin_metrics()
        update_metrics_meta(project_file_structure, metrics)

    return json.loads(meta_file.read_text(encoding="utf-8"))


def update_metrics_meta(project_file_structure: ProjectFileStructure, updated_meta: dict):
    meta_file_path = project_file_structure.metrics_meta
    meta_file_path.parent.mkdir(exist_ok=True)
    meta_file_path.write_text(json.dumps(updated_meta, indent=2), encoding="utf-8")
