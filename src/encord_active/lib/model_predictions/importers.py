# KITTI Reference: https://docs.nvidia.com/tao/archive/tlt-20/tlt-user-guide/text/preparing_data_input.html

import json
import logging
import re
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from encord import Project as EncordProject
from encord.objects.common import Shape
from encord.objects.ontology_object import Object
from encord.objects.ontology_structure import OntologyStructure
from pandas.errors import EmptyDataError
from PIL import Image
from prisma.models import DataUnit
from prisma.types import DataUnitWhereInput
from tqdm.auto import tqdm

from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.db.predictions import (
    BoundingBox,
    Format,
    ObjectDetection,
    Prediction,
)
from encord_active.lib.labels.object import BoxShapes
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
PNG_FILE_NAME_REGEX = r"^(?P<stem>.*?)\.png$"


def import_mask_predictions_obsolete(
    project: EncordProject,
    data_root: Path,
    cache_dir: Path,
    prediction_writer: PredictionWriter,
    class_map: Optional[Dict[str, int]] = None,
    file_name_regex: str = PNG_FILE_NAME_REGEX,
    du_hash_name_lookup: Callable[[Path], Tuple[str, int]] = None,
):
    """
    Will look for mask
    :param project: The project for which to which the predictions are associated
    :param data_root: The root folder of the segmentation masks
    :param cache_dir: The cache directory of the (potentially) cached label rows, images, etc.
    :param prediction_writer: The writer to store the predictions with.
    :param class_map: A Dictionary which contains ``(featureNodeHash, class_idx)`` pairs, such that if
        a pixel in a mask has value 1 and corresponds to ``featureNodeHash == "abcdef01``, then
        ``class_map["abcdef01"] == 1``.
        If the ``class_map`` is not specified, it will be generated by::

            polygon_classes = sorted({
                    o for o in OntologyStructure.from_dict(project.ontology).objects if o.shape == Shape.POLYGON
                },
                key=lambda x: int(x.id)
            )
            {o.feature_node_hash: i+1 for o in enumerate(polygon_classes) if o.shape == "polygon"}

    :param file_name_regex: A regex pattern to match annotation masks. As default, all pngs in the `data_root`
        will be considered as predictions. Example strings that are matched are::

            "/path/to/data_root/masks/some_name.png"
            "/path/to/data_root/some_name.png"

    :param du_hash_name_lookup: A transform ``f(file_path) -> (data_unit_hash: str, frame: int)`` to translate between
        data units and their associated mask file names. You can omit this argument if you

            #. Store masks with identical filenames to the uploaded files.
            #. (only required for videos) append a frame number for video predictions with two under scores.

        For example, if you have predictions for the video ``my_video.mp4``, then you should store prediction masks as
        ``my_video__0.png``, ``my_video__1.png``, etc.

        If you, on the other hand, store files with other naming conventions, we need to be able to match files with
        specific data units (and frames if videos). For example, you may store predictions in folder structures like::

            predictions
            ├── 00115807-3206-4301-b513-026de3f2015e  # data unit hash
            │   ├── 0.png  # Frame number.
            │   └── 1.png
            └── 001722b8-1624-4169-bb11-5977127dcea9
                ├── 0.png
                └── 1.png

        You will then need to specify a lambda function like the following::

            du_hash_name_lookup = lambda file_pth: (file_pth.parent.name, int(file_pth.name))


    """
    label_rows = Project(cache_dir).from_encord_project(project).label_rows
    du_hash_lookup: Dict[str, Tuple[str, int]] = {}

    if not class_map:
        ontology = OntologyStructure.from_dict(project.ontology)
        ontology_polygons: List[Object] = [o for o in ontology.objects if o.shape == Shape.POLYGON]
        class_map = {o.feature_node_hash: i + 1 for i, o in enumerate(ontology_polygons)}

    ontology_hash_lookup: Dict[int, str] = {v: k for k, v in class_map.items()}

    if du_hash_name_lookup is None:
        du_hash_lookup = {}
        for label_row in label_rows.values():
            for du in label_row["data_units"].values():
                fname = du["data_title"].rsplit(".", 1)[0]  # File name without the extension
                if isinstance(du["labels"], list):  # Videos
                    for frame_num in du["labels"]:
                        du_hash_lookup[f"{fname}__{frame_num:05d}"] = du["data_hash"], int(frame_num)
                else:  # Images
                    du_hash_lookup[fname] = du["data_hash"], 0

    queue = [data_root]
    with tqdm(total=None, desc="Importing masks", leave=False) as pbar:
        while queue:
            pth = queue.pop()
            if pth.is_dir():
                queue += list(pth.iterdir())
                continue
            elif pth.is_file():
                match = re.match(file_name_regex, pth.as_posix())
                if not match:
                    continue

                if du_hash_name_lookup:
                    du_hash, frame = du_hash_name_lookup(pth)
                else:
                    du_hash, frame = du_hash_lookup.get(pth.stem, ("", -1))

                if not du_hash:
                    logger.warning(f"Couldn't match file {pth} to any data unit.")
                    continue

                try:
                    input_mask = np.array(Image.open(pth))
                except Exception:
                    continue
                for cls in np.unique(input_mask):
                    if cls == 0:  # background
                        continue

                    class_hash = ontology_hash_lookup[cls]

                    mask = np.zeros_like(input_mask)
                    mask[input_mask == cls] = 1

                    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

                    for contour in contours:
                        if len(contour) < 3 or cv2.contourArea(contour) < 4:
                            continue
                        _mask = np.zeros_like(mask)
                        _mask = cv2.fillPoly(_mask, [contour], 1)

                        prediction_writer.add_prediction(
                            Prediction(
                                data_hash=du_hash,
                                confidence=1.0,
                                object=ObjectDetection(
                                    format=Format.POLYGON,
                                    data=_mask,
                                    feature_hash=class_hash,
                                ),
                            )
                        )

                pbar.update(1)


def migrate_mask_predictions(
    project_dir: Path,
    predictions_dir: Path,
    ontology_mapping: dict[str, int],
    file_name_regex: str = PNG_FILE_NAME_REGEX,
    du_hash_name_lookup: Callable[[Path], Tuple[str, int]] = None,
):
    label_rows = Project(project_dir).load().label_rows
    du_hash_lookup: Dict[str, Tuple[str, int]] = {}
    ontology_hash_lookup: Dict[int, str] = {v: k for k, v in ontology_mapping.items()}

    if du_hash_name_lookup is None:
        du_hash_lookup = {}
        for label_row in label_rows.values():
            for du in label_row["data_units"].values():
                fname = du["data_title"].rsplit(".", 1)[0]  # File name without the extension
                if isinstance(du["labels"], list):  # Videos
                    for frame_num in du["labels"]:
                        du_hash_lookup[f"{fname}__{frame_num:05d}"] = (du["data_hash"], int(frame_num))
                else:  # Images
                    du_hash_lookup[fname] = (du["data_hash"], 0)

    pattern = re.compile(file_name_regex)
    predictions = []
    with tqdm(total=None, desc="Migrating mask predictions", leave=False) as pbar:
        for file_path in predictions_dir.glob("**/*"):
            if not file_path.is_file():
                continue

            match = pattern.match(file_path.name)
            if not match:
                continue

            if du_hash_name_lookup:
                du_hash, frame = du_hash_name_lookup(file_path)
            else:
                du_hash, frame = du_hash_lookup.get(file_path.stem, ("", -1))

            if not du_hash:
                logger.warning(f"Couldn't match file {file_path} to any data unit.")
                continue

            try:
                input_mask = np.asarray(Image.open(file_path))
            except Exception:
                continue
            for cls in np.unique(input_mask):
                if cls == 0:  # background
                    continue

                class_hash = ontology_hash_lookup[cls]
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
                            data_hash=du_hash,
                            confidence=1.0,
                            object=ObjectDetection(format=Format.POLYGON, data=_mask, feature_hash=class_hash),
                        )
                    )
            pbar.update(1)
    return predictions


def import_mask_predictions(
    project: Project,
    predictions_dir: Path,
    ontology_mapping: dict[str, int],
    file_name_regex: str = PNG_FILE_NAME_REGEX,
    du_hash_name_lookup: Callable[[Path], Tuple[str, int]] = None,
):
    predictions = migrate_mask_predictions(
        project.file_structure.project_dir,
        predictions_dir,
        ontology_mapping,
        file_name_regex,
        du_hash_name_lookup,
    )
    import_predictions(project, predictions)


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
        from the `ontology_mapping.json` file located in the predictions directory.
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
        ontology_mapping = _retrieve_ontology_mapping(predictions_dir)

    # Invert the ontology mapping (now it's from object class names to ontology object hashes)
    class_name_to_ontology_hash = {v: k for k, v in ontology_mapping.items()}

    # Check that the input ontology object hashes are valid (exist and represent bounding boxes)
    pfs = ProjectFileStructure(project_dir)
    relevant_ontology_hashes = {
        o.feature_node_hash
        for o in OntologyStructure.from_dict(json.loads(pfs.ontology.read_text())).objects
        if o.shape.value in BoxShapes
    }
    bad_hashes = [h for h in ontology_mapping.keys() if h not in relevant_ontology_hashes]
    if len(bad_hashes) > 0:
        raise ValueError(f"The ontology mapping contains references to invalid ontology object hashes: {bad_hashes}")

    # Migrate predictions from the KITTI format to the Prediction class format
    predictions = []
    for file_path in tqdm(file_paths, desc="Migrating KITTI predictions"):
        # Identify the data unit that corresponds to the given file
        du = _get_matching_data_unit(pfs, file_path, file_path_to_data_unit_func, file_name_regex)
        if du is None:
            continue

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
        from the `ontology_mapping.json` file located in the predictions directory.
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


def import_predictions(project: Project, predictions: List[Prediction]):
    if len(predictions) == 0:
        return

    with PredictionWriter(project) as writer:
        for pred in predictions:
            writer.add_prediction(pred)

    if predictions[0].classification:
        prediction_type = MainPredictionType.CLASSIFICATION
    elif predictions[0].object:
        prediction_type = MainPredictionType.OBJECT
    else:
        raise EmptyDataError("Predictions do not exist!")

    run_all_prediction_metrics(
        data_dir=project.file_structure.project_dir,
        iterator_cls=PredictionIterator,
        use_cache_only=True,
        prediction_type=prediction_type,
    )


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
        logger.info(f'No data unit found for file "{file_path.as_posix()}". Expected exactly one match. Skipping.')
    else:
        logger.info(
            f'Multiple data units found for file "{file_path.as_posix()}". Expected exactly one match. Skipping.'
        )
    return None


def _get_data_unit_identifier(file_path: Path, file_name_regex: str) -> tuple[Optional[str], Optional[int]]:
    match = re.match(file_name_regex, file_path.name)
    if match is None:
        return None, None

    # Obtain 'data_unit_title' and 'frame' named groups to identify the proper data unit
    groups = match.groupdict()
    data_title: Optional[str] = groups.get("data_unit_title")
    frame: Optional[int] = int(groups["frame"]) if "frame" in groups else None

    return data_title, frame


def _retrieve_ontology_mapping(predictions_dir: Path):
    ontology_mapping_file = predictions_dir / "ontology_mapping.json"
    if not ontology_mapping_file.exists():
        raise FileNotFoundError(f'Expected "ontology_mapping.json" file in "{predictions_dir}".')
    ontology_mapping = json.loads(ontology_mapping_file.read_text(encoding="utf-8"))
    return ontology_mapping
