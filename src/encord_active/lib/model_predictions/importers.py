# KITTI Reference: https://docs.nvidia.com/tao/archive/tlt-20/tlt-user-guide/text/preparing_data_input.html

import json
import logging
import re
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import pandas as pd
import PIL
from encord.objects.common import Shape
from encord.objects.ontology_structure import OntologyStructure
from pandas.errors import EmptyDataError
from prisma.models import DataUnit
from prisma.types import DataUnitWhereInput
from tqdm.auto import tqdm

from encord_active.lib.coco.parsers import parse_results
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.db.predictions import (
    BoundingBox,
    Format,
    ObjectDetection,
    Prediction,
)
from encord_active.lib.labels.object import BoxShapes, ObjectShape
from encord_active.lib.metrics.execute import run_all_prediction_metrics
from encord_active.lib.model_predictions.iterator import PredictionIterator
from encord_active.lib.model_predictions.writer import (
    MainPredictionType,
    PredictionWriter,
)
from encord_active.lib.project import Project, ProjectFileStructure

logger = logging.getLogger(__name__)
KITTI_COLUMNS = [
    ("class_name", str),
    ("truncation", float),  # [0, 1]
    ("occlusion", int),  # [0, 3], [ 0 = fully visible, 1 = partly visible, 2 = largely occluded, 3 = unknown]
    ("alpha", float),  # [-pi, pi]
    # bbox
    ("xmin", float),  # [0, img_width]
    ("ymin", float),  # [0, img_height]
    ("xmax", float),  # [0, img_width]
    ("ymax", float),  # [0, img_height]
    # 3-D dimensions (in meters)
    ("height", float),
    ("width", float),
    ("length", float),
    # 3-D object location x, y, z in camera coordinates (in meters)
    ("location_x", float),
    ("location_y", float),
    ("location_z", float),
    # Rotation ry around the Y-axis in camera coordinates
    ("rotation_y", float),  # [-pi, pi]
]

KITTI_FILE_NAME_REGEX = r"^(?P<data_unit_title>.*?)(?:__(?P<frame>\d+))?\.(?:txt|csv)$"
PNG_FILE_NAME_REGEX = r"^(?P<data_unit_title>.*?)(?:__(?P<frame>\d+))?\.png$"


def import_predictions(project: Project, predictions: list[Prediction]):
    """
    Import predictions from Encord's Prediction class format into the specified Encord Active project.

    :param project: The project to import the predictions into.
    :param predictions: The predictions in Encord's Prediction class format to be imported.

    :raises TypeError: If the predictions contain mixed types (both objects and classifications).

    **Note**: All predictions must be of the same type (either objects or classifications).
    Mixing types will raise a `TypeError`.
    """
    if len(predictions) == 0:
        return

    # Ensure that all predictions are of the same type (either objects or classifications) due to structural limitations
    all_object_type = all(pred.object is not None for pred in predictions)
    all_classification_type = all(pred.classification is not None for pred in predictions)

    if all_object_type:
        prediction_type = MainPredictionType.OBJECT
    elif all_classification_type:
        prediction_type = MainPredictionType.CLASSIFICATION
    else:
        raise TypeError("Mismatched prediction types. All predictions must be either objects or classifications")

    with PredictionWriter(project) as writer:
        for pred in predictions:
            writer.add_prediction(pred)

    run_all_prediction_metrics(
        data_dir=project.file_structure.project_dir,
        iterator_cls=PredictionIterator,
        use_cache_only=True,
        prediction_type=prediction_type,
    )


def import_coco_predictions(
    project: Project,
    predictions_json: Path,
    ontology_mapping: Optional[dict[str, int]] = None,
    image_mapping: Optional[dict[int, str]] = None,
):
    """
    Import predictions from the COCO Results format into the specified Encord Active project.

    :param project: The project to import the predictions into.
    :param predictions_json: The JSON file containing the predictions.
    :param ontology_mapping: The mapping from Encord's ontology object hashes to the ids of the COCO categories.
        This mapping allows for the conversion of COCO categories to their corresponding Encord ontology objects.
        It is a dictionary where the keys are the ontology object hashes used in the ontology of the project,
        and the values are the corresponding names of the label classes. It's possible for two ontology objects
        to match the same category if they respectively cover the annotations with polygon or bounding box shape
        of that class.
        If `ontology_mapping` is not specified, the function will attempt to load the mapping
        from the `ontology_mapping.json` file located in the predictions' directory.
        If such file doesn't exist, a `FileNotFoundError` will be raised.
    :param image_mapping: The mapping from the ids of the images in the COCO file to Encord's data unit hashes.
        This mapping allows to accurately match each image mentioned in the COCO results file with its corresponding
        data unit in Encord. It is a dictionary where the keys are the ids of the images in the COCO file, and the
        values are the hashes of the corresponding data units.
        If `image_mapping` is not specified, the function will attempt to load the mapping
        from the `image_mapping.json` file located in the predictions' directory.
        If such file doesn't exist, a `FileNotFoundError` will be raised.
    """
    predictions = migrate_coco_predictions(
        project.file_structure.project_dir,
        predictions_json,
        ontology_mapping,
        image_mapping,
    )
    import_predictions(project, predictions)


def import_kitti_predictions(
    project: Project,
    predictions_dir: Path,
    ontology_mapping: Optional[dict[str, str]] = None,
    file_name_regex: str = KITTI_FILE_NAME_REGEX,
    file_path_to_data_unit_func: Callable[[Path], tuple[str, Optional[int]]] = None,
):
    """
    Import predictions from the KITTI label format into the specified Encord Active project.

    :param project: The project to import the predictions into.
    :param predictions_dir: The directory containing the predictions, including subfolders.
    :param ontology_mapping: The mapping from Encord's ontology object hashes to the names of the label classes.
        This mapping allows for the conversion of label classes to their corresponding Encord ontology objects.
        It is a dictionary where the keys are the ontology object hashes used in the ontology of the project,
        and the values are the corresponding names of the label classes.
        If `ontology_mapping` is not specified, the function will attempt to load the mapping
        from the `ontology_mapping.json` file located in the predictions' directory.
        If such file doesn't exist, a `FileNotFoundError` will be raised.
    :param file_name_regex: A regular expression pattern used to filter the files based on their names.
        Only the files whose names match the pattern will be considered for import.
        Defaults to KITTI_FILE_NAME_REGEX.
    :param file_path_to_data_unit_func: A function to retrieve the data unit hash and optional frame number
        from the file name in order to uniquely identify the data unit.
        If `file_path_to_data_unit_func` is not specified, the function will attempt to retrieve the data unit title
        and optional frame number from the file name using the pattern specified in KITTI_FILE_NAME_REGEX.
        For example, the data unit corresponding to a file with the name `example_image.jpg__5.txt` would have
        the title `example_image.jpg` and it would represent the fifth frame of the image sequence.
    """
    predictions = migrate_kitti_predictions(
        project.file_structure.project_dir,
        predictions_dir,
        ontology_mapping,
        file_name_regex,
        file_path_to_data_unit_func,
    )
    import_predictions(project, predictions)


def import_mask_predictions(
    project: Project,
    predictions_dir: Path,
    ontology_mapping: Optional[dict[str, int]] = None,
    file_name_regex: str = PNG_FILE_NAME_REGEX,
    file_path_to_data_unit_func: Optional[Callable[[Path], tuple[str, Optional[int]]]] = None,
):
    """
    Import predictions from segmentation masks into the specified Encord Active project.

    :param project: The project to import the predictions into.
    :param predictions_dir: The directory containing the predictions, including subfolders.
    :param ontology_mapping: The mapping from Encord's ontology object hashes to the positive integer ids
        of the label classes.
        This mapping allows for the conversion of label classes to their corresponding Encord ontology objects.
        It is a dictionary where the keys are the ontology object hashes used in the ontology of the project,
        and the values are the corresponding ids of the label classes.
        If `ontology_mapping` is not specified, the function will attempt to load the mapping
        from the `ontology_mapping.json` file located in the predictions' directory.
        If such file doesn't exist, a `FileNotFoundError` will be raised.
    :param file_name_regex: A regular expression pattern used to filter the files based on their names.
        Only the files whose names match the pattern will be considered for import.
        Defaults to PNG_FILE_NAME_REGEX.
    :param file_path_to_data_unit_func: A function to retrieve the data unit hash and optional frame number
        from the file name in order to uniquely identify the data unit.
        If `file_path_to_data_unit_func` is not specified, the function will attempt to retrieve the data unit title
        and optional frame number from the file name using the pattern specified in PNG_FILE_NAME_REGEX.
        For example, the data unit corresponding to a file with the name `example_image.jpg__5.png` would have
        the title `example_image.jpg` and it would represent the fifth frame of the image sequence.
    """
    predictions = migrate_mask_predictions(
        project.file_structure.project_dir,
        predictions_dir,
        ontology_mapping,
        file_name_regex,
        file_path_to_data_unit_func,
    )
    import_predictions(project, predictions)


def migrate_coco_predictions(
    project_dir: Path,
    predictions_json: Path,
    ontology_mapping: Optional[dict[str, int]] = None,
    image_mapping: Optional[dict[int, str]] = None,
):
    """
    Migrate predictions from the COCO Results format into the specified Encord Active project.

    :param project_dir: The Encord Active project directory.
    :param predictions_json: The JSON file containing the predictions.
    :param ontology_mapping: The mapping from Encord's ontology object hashes to the ids of the COCO categories.
        This mapping allows for the conversion of COCO categories to their corresponding Encord ontology objects.
        It is a dictionary where the keys are the ontology object hashes used in the ontology of the project,
        and the values are the corresponding names of the label classes. It's possible for two ontology objects
        to match the same category if they respectively cover the annotations with polygon or bounding box shape
        of that class.
        If `ontology_mapping` is not specified, the function will attempt to load the mapping
        from the `ontology_mapping.json` file located in the predictions' directory.
        If such file doesn't exist, a `FileNotFoundError` will be raised.
    :param image_mapping: The mapping from the ids of the images in the COCO file to Encord's data unit hashes.
        This mapping allows to accurately match each image mentioned in the COCO results file with its corresponding
        data unit in Encord. It is a dictionary where the keys are the ids of the images in the COCO file, and the
        values are the hashes of the corresponding data units.
        If `image_mapping` is not specified, the function will attempt to load the mapping
        from the `image_mapping.json` file located in the predictions' directory.
        If such file doesn't exist, a `FileNotFoundError` will be raised.
    :return: The migrated predictions in Encord's Prediction class format.
    """
    # Obtain the predictions contained in the file
    coco_results = parse_results(_json_load(predictions_json))

    # Retrieve the mapping from ontology object hashes to object class ids
    if ontology_mapping is None:
        ontology_mapping = _json_load(predictions_json.parent / "ontology_mapping.json")

    # Retrieve the mapping from COCO image ids to data unit hashes
    if image_mapping is None:
        image_mapping = _json_load(predictions_json.parent / "image_mapping.json")

    # Verify the validity of the ontology object hashes, ensuring they exist and represent bounding boxes
    pfs = ProjectFileStructure(project_dir)
    relevant_ontology_objects = {
        o.feature_node_hash: o for o in OntologyStructure.from_dict(json.loads(pfs.ontology.read_text())).objects
    }
    bad_hashes = [h for h in ontology_mapping.keys() if h not in relevant_ontology_objects]
    if len(bad_hashes) > 0:
        bad_hashes_str = ", ".join(f"'{_}'" for _ in bad_hashes)
        relevant_objects_str = ", ".join(f"{hash_}: '{obj.name}'" for hash_, obj in relevant_ontology_objects.items())
        raise ValueError(
            f"The ontology mapping contains references to invalid ontology object hashes: {bad_hashes_str}.\n"
            "Please, update the mapping with the correct object hashes from the project's ontology.\n"
            f"Use the following ontology objects for references: {relevant_objects_str}."
        )

    # Invert the ontology mapping keeping the target shapes (from object class id + shape to ontology object hash)
    class_id_and_shape_to_ontology_hash = {
        (v, relevant_ontology_objects[k].shape): k for k, v in ontology_mapping.items()
    }

    # Migrate predictions from COCO Results format to the Prediction class format
    predictions = []
    for res in coco_results:
        # Identify the data unit that corresponds to the given file
        du_hash = image_mapping[res.image_id]
        dus = pfs.data_units(DataUnitWhereInput(data_hash=du_hash))
        if len(dus) == 0:
            logger.info(f'No data unit found for data_hash "{du_hash}". Skipping.')
            continue
        du = dus[0]

        # Normalize the prediction's points by their image size to match the Encord format
        if res.segmentation:
            format_, shape = Format.POLYGON, Shape.POLYGON
            data = res.segmentation / np.array([[du.width, du.height]])
        elif res.bbox:
            format_, shape = Format.BOUNDING_BOX, Shape.BOUNDING_BOX
            orig_x, orig_y, orig_w, orig_h = res.bbox
            x = orig_x / du.width
            y = orig_y / du.height
            w = orig_w / du.width
            h = orig_h / du.height
            data = BoundingBox(x=x, y=y, w=w, h=h)
        else:
            raise Exception(f'Unsupported format found in the prediction "{res}". Expected a bounding box or polygon.')

        ontology_obj_hash = class_id_and_shape_to_ontology_hash.get((res.category_id, shape))
        if ontology_obj_hash is None:
            logger.info(f'No ontology object found for category id "{res.category_id}" and shape "{shape}". Skipping.')
            continue

        predictions.append(
            Prediction(
                data_hash=du.data_hash,
                confidence=res.score,
                object=ObjectDetection(
                    feature_hash=ontology_obj_hash,
                    format=format_,
                    data=data,
                ),
            )
        )

    return predictions


def migrate_kitti_predictions(
    project_dir: Path,
    predictions_dir: Path,
    ontology_mapping: Optional[dict[str, str]] = None,
    file_name_regex: str = KITTI_FILE_NAME_REGEX,
    file_path_to_data_unit_func: Optional[Callable[[Path], tuple[str, Optional[int]]]] = None,
) -> list[Prediction]:
    """
    Migrate predictions from the KITTI label format to Encord's Prediction class format.

    :param project_dir: The Encord Active project directory.
    :param predictions_dir: The directory containing the predictions, including subfolders.
    :param ontology_mapping: The mapping from Encord's ontology object hashes to the names of the label classes.
        This mapping allows for the conversion of label classes to their corresponding Encord ontology objects.
        It is a dictionary where the keys are the ontology object hashes used in the ontology of the project,
        and the values are the corresponding names of the label classes.
        If `ontology_mapping` is not specified, the function will attempt to load the mapping
        from the `ontology_mapping.json` file located in the predictions' directory.
        If such file doesn't exist, a `FileNotFoundError` will be raised.
    :param file_name_regex: A regular expression pattern used to filter the files based on their names.
        Only the files whose names match the pattern will be considered for migration.
        Defaults to KITTI_FILE_NAME_REGEX.
    :param file_path_to_data_unit_func: A function to retrieve the data unit hash and optional frame number
        from the file name in order to uniquely identify the data unit.
        If `file_path_to_data_unit_func` is not specified, the function will attempt to retrieve the data unit title
        and optional frame number from the file name using the pattern specified in KITTI_FILE_NAME_REGEX.
        For example, the data unit corresponding to a file with the name `example_image.jpg__5.txt` would have
        the title `example_image.jpg` and it would represent the fifth frame of the image sequence.
    :return: The migrated predictions in Encord's Prediction class format.
    """

    # Obtain the files containing predictions
    pattern = re.compile(file_name_regex)
    file_paths = [
        path
        for path in predictions_dir.glob("**/*")
        if path.is_file() and path.suffix.lower() in [".txt", ".csv"] and pattern.match(path.name) is not None
    ]

    # Retrieve the mapping from ontology object hashes to label names
    if ontology_mapping is None:
        ontology_mapping = _json_load(predictions_dir / "ontology_mapping.json")
    # Invert the ontology mapping (now it's from object class names to ontology object hashes)
    class_name_to_ontology_hash = {v: k for k, v in ontology_mapping.items()}

    # Verify the validity of the ontology object hashes, ensuring they exist and represent bounding boxes
    pfs = ProjectFileStructure(project_dir)
    relevant_ontology_objects = {
        o.feature_node_hash: o
        for o in OntologyStructure.from_dict(json.loads(pfs.ontology.read_text())).objects
        if o.shape.value in BoxShapes
    }
    bad_hashes = [h for h in ontology_mapping.keys() if h not in relevant_ontology_objects]
    if len(bad_hashes) > 0:
        bad_hashes_str = ", ".join(f"'{_}'" for _ in bad_hashes)
        relevant_objects_str = ", ".join(f"{hash_}: '{obj.name}'" for hash_, obj in relevant_ontology_objects.items())
        raise ValueError(
            f"The ontology mapping contains references to invalid ontology object hashes: {bad_hashes_str}.\n"
            "Please, update the mapping with the correct object hashes from the project's ontology.\n"
            f"Use the following ontology objects for references: {relevant_objects_str}."
        )

    # Migrate predictions from the KITTI format to the Prediction class format
    predictions = []
    for file_path in tqdm(file_paths, desc="Migrating KITTI predictions"):
        # Identify the data unit that corresponds to the given file
        du = _get_matching_data_unit(pfs, file_path, file_path_to_data_unit_func, file_name_regex)
        if du is None:
            continue

        # Read the predictions
        try:
            df = pd.read_csv(file_path, sep=" ", header=None)
        except EmptyDataError:
            continue

        # Include headers and account for additional "custom" columns
        headers = list(map(lambda _tuple: _tuple[0], KITTI_COLUMNS))
        headers += [f"undefined_{i}" for i in range(df.shape[1] - len(headers))]
        df.columns = pd.Index(headers)

        # Add KITTI bounding boxes predictions
        for row in df.to_dict("records"):
            ontology_obj_hash = class_name_to_ontology_hash[row["class_name"]]

            # Normalize bounding box dimensions by their image size to match the Encord format
            x = row["xmin"] / du.width
            y = row["ymin"] / du.height
            w = (row["xmax"] - row["xmin"]) / du.width
            h = (row["ymax"] - row["ymin"]) / du.height
            bbox = BoundingBox(x=x, y=y, w=w, h=h)

            predictions.append(
                Prediction(
                    data_hash=du.data_hash,
                    confidence=float(row["undefined_0"]),
                    object=ObjectDetection(format=Format.BOUNDING_BOX, data=bbox, feature_hash=ontology_obj_hash),
                )
            )
    return predictions


def migrate_mask_predictions(
    project_dir: Path,
    predictions_dir: Path,
    ontology_mapping: Optional[dict[str, int]] = None,
    file_name_regex: str = PNG_FILE_NAME_REGEX,
    file_path_to_data_unit_func: Optional[Callable[[Path], tuple[str, Optional[int]]]] = None,
) -> list[Prediction]:
    """
    Migrate predictions from segmentation masks into the specified Encord Active project.

    :param project_dir: The Encord Active project directory.
    :param predictions_dir: The directory containing the predictions, including subfolders.
    :param ontology_mapping: The mapping from Encord's ontology object hashes to the positive integer ids
        of the label classes.
        This mapping allows for the conversion of label classes to their corresponding Encord ontology objects.
        It is a dictionary where the keys are the ontology object hashes used in the ontology of the project,
        and the values are the corresponding ids of the label classes.
        If `ontology_mapping` is not specified, the function will attempt to load the mapping
        from the `ontology_mapping.json` file located in the predictions' directory.
        If such file doesn't exist, a `FileNotFoundError` will be raised.
    :param file_name_regex: A regular expression pattern used to filter the files based on their names.
        Only the files whose names match the pattern will be considered for migration.
        Defaults to PNG_FILE_NAME_REGEX.
    :param file_path_to_data_unit_func: A function to retrieve the data unit hash and optional frame number
        from the file name in order to uniquely identify the data unit.
        If `file_path_to_data_unit_func` is not specified, the function will attempt to retrieve the data unit title
        and optional frame number from the file name using the pattern specified in PNG_FILE_NAME_REGEX.
        For example, the data unit corresponding to a file with the name ``example_image.jpg__5.png`` would have
        the title `example_image.jpg` and it would represent the fifth frame of the image sequence.
    :return: The migrated predictions in Encord's Prediction class format.
    """
    # Obtain the files containing predictions
    pattern = re.compile(file_name_regex)
    file_paths = [
        path
        for path in predictions_dir.glob("**/*")
        if path.is_file() and path.suffix.lower() in [".json"] and pattern.match(path.name) is not None
    ]

    # Retrieve the mapping from ontology object hashes to object class ids
    if ontology_mapping is None:
        ontology_mapping = _json_load(predictions_dir / "ontology_mapping.json")
    # Invert the ontology mapping (now it's from object class ids to ontology object hashes)
    class_id_to_ontology_hash = {v: k for k, v in ontology_mapping.items()}

    # Verify the validity of the ontology object hashes, ensuring they exist and represent polygons
    pfs = ProjectFileStructure(project_dir)
    relevant_ontology_objects = {
        o.feature_node_hash: o
        for o in OntologyStructure.from_dict(json.loads(pfs.ontology.read_text())).objects
        if o.shape.value == ObjectShape.POLYGON
    }
    bad_hashes = [h for h in ontology_mapping.keys() if h not in relevant_ontology_objects]
    if len(bad_hashes) > 0:
        bad_hashes_str = ", ".join(f"'{_}'" for _ in bad_hashes)
        relevant_objects_str = ", ".join(f"{hash_}: '{obj.name}'" for hash_, obj in relevant_ontology_objects.items())
        raise ValueError(
            f"The ontology mapping contains references to invalid ontology object hashes: {bad_hashes_str}.\n"
            "Please, update the mapping with the correct object hashes from the project's ontology.\n"
            f"Use the following ontology objects for references: {relevant_objects_str}."
        )

    # Migrate predictions from segmentation masks to the Prediction class format
    predictions = []
    for file_path in tqdm(file_paths, desc="Migrating mask predictions", leave=False):
        # Identify the data unit that corresponds to the given file
        du = _get_matching_data_unit(pfs, file_path, file_path_to_data_unit_func, file_name_regex)
        if du is None:
            continue

        # Read the predictions
        try:
            input_mask = np.asarray(PIL.Image.open(file_path))
        except PIL.UnidentifiedImageError:
            continue

        # Ensure that the sizes of the mask and the original image are identical
        pass

        # Include predictions for each class identified in the mask
        for cls in np.unique(input_mask):
            if cls == 0:  # Ignore background
                continue

            ontology_obj_hash = class_id_to_ontology_hash[cls]
            mask = np.zeros_like(input_mask)
            mask[input_mask == cls] = 1
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

            for contour in contours:
                if len(contour) < 3 or cv2.contourArea(contour) < 4:
                    continue
                _mask = np.zeros_like(mask)
                _mask = cv2.fillPoly(_mask, [contour], 1)

                predictions.append(
                    Prediction(
                        data_hash=du.data_hash,
                        confidence=1.0,
                        object=ObjectDetection(format=Format.POLYGON, data=_mask, feature_hash=ontology_obj_hash),
                    )
                )
    return predictions


# ================================= UTILITY FUNCTIONS =================================


def _get_data_unit_identifier(file_path: Path, file_name_regex: str) -> tuple[Optional[str], Optional[int]]:
    match = re.match(file_name_regex, file_path.name)
    if match is None:
        return None, None

    # Obtain 'data_unit_title' and 'frame' named groups to identify the proper data unit
    groups = match.groupdict()
    data_title: Optional[str] = groups.get("data_unit_title")
    frame: Optional[int] = int(groups["frame"]) if "frame" in groups else None

    return data_title, frame


def _get_matching_data_unit(
    pfs: ProjectFileStructure,
    file_path: Path,
    file_path_to_data_unit_func: Optional[Callable[[Path], tuple[str, Optional[int]]]],
    file_name_regex: str,
) -> Optional[DataUnit]:
    where_arg: DataUnitWhereInput
    if file_path_to_data_unit_func is None:
        # If the 'file_path_to_data_unit_func' function is missing, find data unit title
        # and optional frame indicator within the file name as specified in the regex param
        data_title, frame = _get_data_unit_identifier(file_path, file_name_regex)
        if data_title is None:  # Skip file if its name doesn't match the regex
            return None
        where_arg = (
            DataUnitWhereInput(data_title=data_title)
            if frame is None
            else DataUnitWhereInput(data_title=data_title, frame=frame)
        )
    else:
        data_hash, frame = file_path_to_data_unit_func(file_path)
        where_arg = (
            DataUnitWhereInput(data_hash=data_hash)
            if frame is None
            else DataUnitWhereInput(data_hash=data_hash, frame=frame)
        )

    # Search for the only data unit that matches the input field values, return 'None' in case of nonexistence/ambiguity
    with PrismaConnection(pfs) as conn:
        data_units = conn.dataunit.find_many(where=where_arg, take=2)
    if len(data_units) == 1:
        return data_units[0]
    if len(data_units) == 0:
        logger.info(f'No data unit found for file "{file_path}". Expected exactly one match. Skipping.')
    else:
        logger.info(f'Multiple data units found for file "{file_path}". Expected exactly one match. Skipping.')
    return None


def _json_load(mapping_file: Path):
    if not mapping_file.exists() or not mapping_file.is_file() or mapping_file.suffix != ".json":
        raise FileNotFoundError(f'JSON file with expected path "{mapping_file}" was not found')
    mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
    return mapping
