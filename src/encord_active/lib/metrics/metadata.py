import json
from pathlib import Path


def fetch_metrics_meta(project_dir: Path):
    meta_file = project_dir / "metrics" / "metrics_meta.json"
    if not meta_file.is_file():
        # todo raise an exception after a few releases when it's a rule to have such file
        return dict()

    return json.loads(meta_file.read_text(encoding="utf-8"))


def update_metrics_meta(project_dir: Path, updated_meta: dict):
    meta_file_path = project_dir / "metrics" / "metrics_meta.json"
    meta_file_path.parent.mkdir(exist_ok=True)
    meta_file_path.write_text(json.dumps(updated_meta, indent=2), encoding="utf-8")
