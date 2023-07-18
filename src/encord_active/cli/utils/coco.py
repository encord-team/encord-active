import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict

import numpy as np
from encord.objects.common import Shape

from encord_active.cli.config import app_config
from encord_active.lib.coco.importer import CocoImporter
from encord_active.lib.coco.parsers import parse_results
from encord_active.lib.db.predictions import (
    BoundingBox,
    Format,
    ObjectDetection,
    Prediction,
)
from encord_active.lib.metrics.execute import run_metrics
from encord_active.lib.project import ProjectFileStructure


def import_coco_predictions(
    project_path: Path,
    predictions_path: Path,
):
    project_file_structure = ProjectFileStructure(project_path)
    image_data_unit = json.loads(project_file_structure.image_data_unit.read_text(encoding="utf-8"))
    ontology = json.loads(project_file_structure.ontology.read_text(encoding="utf-8"))
    # NOTE: when we import a coco project, we change the category id to support
    # categories with multiple shapes. Here we iterate the ontology objects
    # and the ids are in the following format:
    # obj["id"] == `10` ->
    #   1 - original category_id
    #   0 - index of the shape -- we can discard and use the actual shape
    # NOTE: we subtract 1 from the id to match the original id since we don't
    # support 0 index when the project is created
    category_to_hash = {
        (str(int(obj["id"][:-1]) - 1), obj["shape"]): obj["featureNodeHash"] for obj in ontology["objects"]
    }

    category_types: Dict[int, list] = {}
    for obj in ontology["objects"]:
        category_types.setdefault(int(obj["id"][:-1]) - 1, []).append(obj["shape"])

    predictions = []
    results = json.loads(predictions_path.read_text(encoding="utf-8"))

    coco_results = parse_results(results)
    for res in coco_results:
        image = image_data_unit[str(res.image_id)]
        img_h, img_w = image["height"], image["width"]

        if res.segmentation is not None and res.segmentation:
            format, shape = Format.POLYGON, Shape.POLYGON
            data = res.segmentation / np.array([[img_w, img_h]])
        elif res.bbox:
            format, shape = Format.BOUNDING_BOX, Shape.BOUNDING_BOX
            x, y, w, h = res.bbox

            x = min(max(x / img_w, 0.0), 1.0)
            y = min(max(y / img_h, 0.0), 1.0)

            # Make sure w and h do not cause the bounding box to exceed the limits.
            w = min(max(w / img_w, 0.0), 1.0 - x)
            h = min(max(h / img_h, 0.0), 1.0 - y)

            data = BoundingBox(x=x, y=y, w=w, h=h)
        else:
            raise Exception("Unsupported result format")

        if shape.value not in category_types[res.category_id]:
            raise TypeError(
                f"Found a prediction with shape '{shape}' for the category '{res.category_id}',"
                f"however there is no category with that shape in the ontology."
            )

        predictions.append(
            Prediction(
                data_hash=image["data_hash"],
                confidence=res.score,
                object=ObjectDetection(
                    feature_hash=category_to_hash[(str(res.category_id), shape.value)],
                    format=format,
                    data=data,
                ),
            )
        )

    return predictions


def import_coco_project(images_dir: Path, annotations_file: Path, target: Path, use_symlinks: bool = False) -> Path:
    """
    Importer for COCO datasets.
    Args:
        images_dir (Path): path where images are stored
        annotations_file (Path): the COCO JSON annotation file
        target (Path): where to store the data
        use_symlinks (bool): if False, the importer will copy images.
            Otherwise, symlinks will be used to save disk space.
    """
    coco_importer = CocoImporter(
        images_dir_path=images_dir,
        annotations_file_path=annotations_file,
        destination_dir=target,
        use_symlinks=use_symlinks,
    )

    with TemporaryDirectory() as temp_dir:
        temp_folder = Path(temp_dir)

        dataset = coco_importer.create_dataset(temp_folder)
        ontology = coco_importer.create_ontology()
        coco_importer.create_project(
            dataset=dataset,
            ontology=ontology,
            ssh_key_path=app_config.get_ssh_key(),
        )

    run_metrics(data_dir=coco_importer.project_dir, use_cache_only=True)

    return coco_importer.project_dir
