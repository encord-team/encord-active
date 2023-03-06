from pathlib import Path

import yaml


def fetch_metrics_meta(project_dir: Path):
    meta_file = project_dir / "metrics_meta.yaml"
    if not meta_file.is_file():
        # todo raise an exception after a few releases when it's a rule to have such file
        return dict()

    with meta_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def update_metrics_meta(project_dir: Path, updated_meta):
    meta_file_path = project_dir / "metrics_meta.yaml"
    meta_file_path.write_text(yaml.safe_dump(updated_meta), encoding="utf-8")
