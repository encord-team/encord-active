import json
import pickle
from pathlib import Path
from typing import Optional

import typer

from encord_active.app.cli.config import app_config

import_cli = typer.Typer(rich_markup_mode="markdown")


@import_cli.command(name="predictions")
def import_predictions(
    predictions_path: Path = typer.Argument(
        ...,
        help="Path to a predictions file.",
        dir_okay=False,
    ),
    coco: bool = typer.Option(False, help="Import a coco result format file"),
):
    """
    [bold]Imports[/bold] a predictions file. The predictions should be using the `Prediction` model and be stored in a pkl file.
    If `--coco` option is specified the file should be a json following the coco results format. :brain:
    """

    from encord_active.app.common.cli_helpers import choose_local_project

    project_dir = choose_local_project(app_config)
    if not project_dir:
        raise typer.Abort()

    from encord_active.app.common.cli_helpers import get_local_project

    project = get_local_project(project_dir)

    if coco:
        import numpy as np

        from encord_active.app.db.predictions import BoundingBox, Format, Prediction
        from encord_active.lib.coco.importer import IMAGE_DATA_UNIT_FILENAME
        from encord_active.lib.coco.parsers import parse_results

        image_data_unit = json.loads((Path(project_dir) / IMAGE_DATA_UNIT_FILENAME).read_text(encoding="utf-8"))
        ontology = json.loads((Path(project_dir) / "ontology.json").read_text(encoding="utf-8"))
        category_to_hash = {obj["id"]: obj["featureNodeHash"] for obj in ontology["objects"]}
        predictions = []
        results = json.loads(Path(predictions_path).read_text(encoding="utf-8"))

        coco_results = parse_results(results)
        for res in coco_results:
            image = image_data_unit[str(res.image_id)]
            img_h, img_w = image["height"], image["width"]

            if res.segmentation is not None:
                format = Format.POLYGON
                data = res.segmentation / np.array([[img_h, img_w]])
            elif res.bbox:
                format = Format.BOUNDING_BOX
                x, y, w, h = res.bbox
                data = BoundingBox(x=x / img_w, y=y / img_h, w=w / img_w, h=h / img_h)
            else:
                raise Exception("Unsupported result format")

            predictions.append(
                Prediction(
                    data_hash=image["data_hash"],
                    class_id=category_to_hash[str(res.category_id)],
                    confidence=res.score,
                    format=format,
                    data=data,
                )
            )
    else:
        with open(predictions_path, "rb") as f:
            predictions = pickle.load(f)

    from encord_active.app.db.predictions import (
        import_predictions as app_import_predictions,
    )

    app_import_predictions(project, project_dir, predictions)


@import_cli.command(name="project")
def import_project(
    images_dir: Optional[Path] = typer.Argument(
        None,
        help="Path to the directory containing the dataset images. (only required with `--cooc`)",
        file_okay=False,
        exists=True,
    ),
    annotations_file: Optional[Path] = typer.Argument(
        None,
        help="Path to the file containing the dataset annotations. (only required with `--cooc`)",
        dir_okay=False,
        exists=True,
    ),
    encord_project_hash: Optional[str] = typer.Option(
        None, "--project_hash", help="Encord project hash of the project you wish to import."
    ),
    coco: bool = typer.Option(False, help="Import a project from the coco format"),
):
    """
    [bold]Imports[/bold] a new project from Encord or a local coco project 📦
    """
    if coco:
        from encord_active.lib.coco.importer import CocoImporter
        from encord_active.lib.metrics.run_all import run_metrics

        if not annotations_file:
            raise typer.BadParameter("`annotations_file` argument is missing")
        elif not images_dir:
            raise typer.BadParameter("`images_dir` argument is missing")

        ssh_key_path = app_config.get_or_query_ssh_key()
        projects_root = app_config.get_or_query_project_path()

        coco_importer = CocoImporter(
            images_dir_path=images_dir,
            annotations_file_path=annotations_file,
            ssh_key_path=ssh_key_path,
            destination_dir=projects_root,
        )

        dataset = coco_importer.create_dataset()
        ontology = coco_importer.create_ontology()
        coco_importer.create_project(
            dataset=dataset,
            ontology=ontology,
        )
        run_metrics(data_dir=coco_importer.project_dir)
    else:
        from encord_active.lib.metrics.import_encord_project import (
            main as import_script,
        )

        import_script(app_config, encord_project_hash=encord_project_hash)
