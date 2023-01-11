"""DENIS: think properly about the structure, the transcoders and how we want to subclass or extend this so people
can plug into the different parts easily.
ideas
* a class where the individual parts can be overwritten
* a class where the individual transformers can be re-assigned
*
DENIS: how are we going to document this in Sphinx if this class is independent?
DENIS:
* parallel downloads with a specific flag
* saving the annotation file with a specific flag
* labels class for better type support.
"""
import copy
import datetime
import json
import logging
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pandas as pd
import requests
from encord.objects.common import Shape
from encord.objects.ontology_object import Object
from encord.objects.ontology_structure import OntologyStructure
from shapely.geometry import Polygon
from tqdm.auto import tqdm

from encord_active.lib.coco.datastructure import (
    CocoAnnotation,
    CocoBbox,
    SuperClass,
    to_attributes_field,
)

logger = logging.getLogger(__name__)

DOWNLOAD_FILES_DEFAULT = False
FORCE_DOWNLOAD_DEFAULT = False
DOWNLOAD_FILE_PATH_DEFAULT = Path(".")
INCLUDE_VIDEOS_DEFAULT = True
INCLUDE_UNANNOTATED_VIDEOS_DEFAULT = False
INCLUDE_NULL_ANNOTATIONS = True


@dataclass
class Size:
    width: int
    height: int


@dataclass
class ImageLocation:
    data_hash: str
    file_name: str


@dataclass
class VideoLocation:
    data_hash: str
    file_name: str
    frame_num: int


DataLocation = Union[ImageLocation, VideoLocation]


class EncodingError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def get_size(*args, **kwargs) -> Size:
    # DENIS: this belongs in a utils folder.
    return Size(1, 2)  # A stub for now


def get_polygon_from_dict(polygon_dict, W, H):
    return [(polygon_dict[str(i)]["x"] * W, polygon_dict[str(i)]["y"] * H) for i in range(len(polygon_dict))]


# DENIS: TODO: focus on doing the parser for now for segmentations for images as it was intended. Seems like
#   for other formats I can still add stuff or have the clients extend what we have.

# DENIS: should these labels be the data structure that I've invented for them instead of the encord dict?
class CocoEncoder:
    """This class has been purposefully built in a modular fashion for extensibility in mind. You are encouraged to
    subclass this encoder and change any of the custom functions. The return types are given for convenience, but these
    can also be subclassed for your needs.
    All functions which could be static are deliberately not made static so you have the option to access the `self`
    object at any time.
    """

    def __init__(self, labels_list: List[dict], metrics: dict, ontology: OntologyStructure):
        self._labels_list = labels_list
        self._metrics = metrics
        self._ontology = ontology
        self._coco_json: dict = dict()
        self._current_annotation_id: int = 0
        self._object_hash_to_track_id_map: dict = dict()
        self._coco_categories_id_to_ontology_object_map: dict = dict()  # DENIS: do we need this?
        self._feature_hash_to_coco_category_id_map: dict = dict()
        self._data_hash_to_image_id_map: dict = dict()
        """Map of (data_hash, frame_offset) to the image id"""

        # self._data_location_to_image_id_map = dict()

        self._download_files = DOWNLOAD_FILES_DEFAULT
        self._force_download = FORCE_DOWNLOAD_DEFAULT
        self._download_file_path = DOWNLOAD_FILE_PATH_DEFAULT
        self._include_videos = INCLUDE_VIDEOS_DEFAULT
        self._include_unannotated_videos = INCLUDE_UNANNOTATED_VIDEOS_DEFAULT
        self._include_null_annotations = INCLUDE_NULL_ANNOTATIONS

    def encode(
        self,
        *,
        download_files: bool = DOWNLOAD_FILES_DEFAULT,
        force_download: bool = FORCE_DOWNLOAD_DEFAULT,
        download_file_path: Path = DOWNLOAD_FILE_PATH_DEFAULT,
        include_videos: bool = INCLUDE_VIDEOS_DEFAULT,
        include_unannotated_videos: bool = INCLUDE_UNANNOTATED_VIDEOS_DEFAULT,
        include_null_annotation: bool = INCLUDE_NULL_ANNOTATIONS,
    ) -> dict:
        """
        Args:
            download_files: If set to true, the images are downloaded into a local directory and the `coco_url` of the
                images will point to the location of the local directory. DENIS: can also maybe have a Path obj here.
            force_download: If set to true, the images are downloaded even if they already exist in the local directory.
            download_file_path:
                Root path to where the images and videos are downloaded or where downloaded images are looked up from.
                For example, if `include_unannotated_videos = True` then this is the root path of the
                `videos/<data_hash>` directory.
            include_videos: If set to true, the videos are included in the output.
            include_unannotated_videos:
                This will be ignored if the files are not downloaded (whether they are being downloaded now or they
                were already there) in which case it will default to False. The code will assume that the video is
                downloaded and expanded correctly in the same way that would happen if the video was downloaded via
                the `download_files = True` argument.
            include_null_annotation:
                If set to true, then annotations with null values will still be included in the json file."""
        self._download_files = download_files
        self._force_download = force_download
        self._download_file_path = download_file_path
        self._include_videos = include_videos
        self._include_unannotated_videos = include_unannotated_videos
        self._include_null_annotations = include_null_annotation

        self._coco_json["info"] = self.get_info()
        self._coco_json["categories"] = self.get_categories()
        self._coco_json["images"] = self.get_images()  # TODO: remove images without annotations
        self._coco_json["annotations"] = self.get_annotations()

        return self._coco_json

    def get_info(self) -> dict:
        return {
            "description": self.get_description(),
            "contributor": None,  # TODO: these fields also need a response
            "date_created": str(datetime.datetime.now().isoformat()),
            "url": None,
            "version": None,
            "year": None,
        }

    def get_description(self) -> Optional[str]:
        if len(self._labels_list) == 0:
            return None
        else:
            return self._download_file_path.as_posix().split("/")[-1]

    def get_categories(self) -> List[dict]:
        """This does not translate classifications as they are not part of the Coco spec."""
        categories = []
        for object_ in self._ontology.objects:
            categories.append(self.get_category(object_))

        return categories

    def get_category(self, object_: Object) -> dict:
        super_category = self.get_super_category(object_)
        ret = {
            "supercategory": super_category,
            "id": self.add_to_object_map_and_get_next_id(object_),
            "name": object_.name,
            "object_hash": object_.feature_node_hash,
        }
        if object_.shape.value == "point":
            # TODO: we will have to do something similar for skeletons.
            ret["keypoints"] = "keypoint"
            ret["skeleton"] = []
        return ret

    def get_super_category(self, object_: Object) -> str:
        return object_.shape.value

    def add_to_object_map_and_get_next_id(self, object_: Object) -> int:
        # 0 id is reserved for the background class
        id_ = len(self._coco_categories_id_to_ontology_object_map) + 1
        self._coco_categories_id_to_ontology_object_map[id_] = object_
        self._feature_hash_to_coco_category_id_map[object_.feature_node_hash] = id_
        return id_

    def get_category_name(self, object_: Object) -> str:
        return object_.name

    def get_images(self) -> list:
        """All the data is in the specific label_row"""
        images = []
        for labels in tqdm(self._labels_list, desc="Downloading data units", position=0, leave=True):
            label_hash = labels["label_hash"]
            for data_unit in labels["data_units"].values():
                data_type = data_unit["data_type"]
                if "application/dicom" in data_type:
                    images.extend(self.get_dicom(label_hash, data_unit))
                elif "video" not in data_type:
                    images.append(self.get_image(label_hash, data_unit))
                else:
                    images.extend(self.get_video_images(label_hash, data_unit))
        return images

    def get_dicom(self, label_hash: str, data_unit: dict) -> list:
        # NOTE: could give an option whether to include dicoms, but this is inferred by which labels we request.

        data_hash = data_unit["data_hash"]

        images = []

        height = data_unit["height"]
        width = data_unit["width"]

        for frame_num in data_unit["labels"].keys():
            dicom_image = self.get_dicom_image(label_hash, data_hash, height, width, int(frame_num))
            images.append(dicom_image)

        return images

    def get_image(self, label_hash: str, data_unit: dict) -> dict:
        # DENIS: we probably want a map of this image id to image hash in our DB, including the image_group hash.

        """
        DENIS: next up: here we need to branch off and create the videos
        * coco_url, height, width will be the same
        * id will be continuous
        * file_name will be also continuous according to all the images that are being extracted from the video.
        Do all the frames, and the ones without annotations will just have no corresponding annotations. We can
        still later have an option to exclude them and delete the produced images.
        """
        image_id = len(self._data_hash_to_image_id_map)
        data_hash = data_unit["data_hash"]
        self._data_hash_to_image_id_map[(data_hash, 0)] = image_id
        return {
            "coco_url": data_unit["data_link"],
            "flickr_url": "",
            "id": image_id,
            "image_title": data_unit["data_title"],
            "file_name": self.get_file_name_and_download_image(label_hash, data_unit),
            "height": data_unit["height"],
            "width": data_unit["width"],
            "label_hash": label_hash,
            "data_hash": data_hash,
            "frame_num": int(data_unit["data_sequence"]),
        }

    def get_file_name_and_download_image(self, label_hash: str, data_unit: dict) -> str:
        data_hash = data_unit["data_hash"]
        url = data_unit["data_link"]
        image_extension = data_unit["data_type"].split("/")[-1]

        relative_destination_path = Path("data").joinpath(
            Path(label_hash), Path("images"), Path(f"{data_hash}.{image_extension}")
        )
        absolute_destination_path = self._download_file_path.joinpath(relative_destination_path)
        download_condition = self._download_files and not absolute_destination_path.exists()

        if download_condition or self._force_download:
            self.download_image(url, absolute_destination_path)

        return str(relative_destination_path)

    def get_video_images(self, label_hash: str, data_unit: dict) -> List[dict]:
        if not self._include_videos:
            return []

        video_title = data_unit["data_title"]
        url = data_unit["data_link"]
        data_hash = data_unit["data_hash"]

        destination_path = self._download_file_path.joinpath(Path("data"), Path(label_hash), Path("images"))
        download_condition = self._download_files and len(list(destination_path.glob("*.png"))) == 0
        if download_condition or self._force_download:
            self.download_video_images(url, label_hash, data_hash)

        images = []
        coco_url = data_unit["data_link"]
        height = data_unit["height"]
        width = data_unit["width"]

        path_to_video_dir = self._download_file_path.joinpath(Path("videos"), Path(data_hash))
        if self._include_unannotated_videos and path_to_video_dir.is_dir():
            # DENIS: log something for transparency?
            for frame_num in range(len(list(path_to_video_dir.iterdir()))):
                images.append(
                    self.get_video_image(label_hash, data_hash, video_title, coco_url, height, width, int(frame_num))
                )
        else:
            for frame_num in data_unit["labels"].keys():
                images.append(
                    self.get_video_image(label_hash, data_hash, video_title, coco_url, height, width, int(frame_num))
                )

        return images

    # def get_frame_numbers(self, data_unit: dict) -> Iterator:  # DENIS: use this to remove the above if/else.

    def get_dicom_image(self, label_hash: str, data_hash: str, height: int, width: int, frame_num: int) -> dict:
        image_id = len(self._data_hash_to_image_id_map)
        self._data_hash_to_image_id_map[(data_hash, frame_num)] = image_id

        return {
            # DICOM does not have a one to one mapping between a frame and a DICOM series file.
            "coco_url": "",
            "id": image_id,
            "file_name": self.get_dicom_file_path(data_hash, frame_num),
            "height": height,
            "width": width,
            "label_hash": label_hash,
            "data_hash": data_hash,
            "frame_num": frame_num,
        }

    def get_video_image(
        self, label_hash: str, data_hash: str, video_title: str, coco_url: str, height: int, width: int, frame_num: int
    ):
        image_id = len(self._data_hash_to_image_id_map)
        self._data_hash_to_image_id_map[(data_hash, frame_num)] = image_id

        return {
            "coco_url": coco_url,
            "id": image_id,
            "video_title": video_title,
            "file_name": self.get_video_file_path(data_hash, frame_num),
            "height": height,
            "width": width,
            "label_hash": label_hash,
            "data_hash": data_hash,
            "frame_num": frame_num,
        }

    def get_dicom_file_path(self, data_hash: str, frame_num: int) -> str:
        path = Path("dicom") / data_hash / str(frame_num)
        return str(path)

    def get_video_file_path(self, data_hash: str, frame_num: int) -> str:
        frame_file_name = Path(f"{frame_num}.jpg")
        video_file_path = Path("videos").joinpath(Path(data_hash), frame_file_name)
        return str(video_file_path)

    def download_video_images(self, url: str, data_hash: str, destination_path: Path) -> None:
        with tempfile.TemporaryDirectory(str(Path("."))) as temporary_directory:
            video_location = Path(temporary_directory).joinpath(Path(data_hash))
            download_file(
                url,
                video_location,
            )

            extract_frames(video_location, destination_path, data_hash)

    def get_annotations(self):
        annotations = []

        # DENIS: need to make sure at least one image
        for labels in tqdm(self._labels_list, desc="Processing annotations"):
            label_hash = labels["label_hash"]
            if label_hash not in self._metrics.keys():
                continue
            for data_unit in labels["data_units"].values():
                data_hash = data_unit["data_hash"]
                if data_hash not in self._metrics[label_hash].keys():
                    continue
                data_unit_metrics = self._metrics[label_hash][data_hash]

                if data_unit["data_type"] in ["video", "application/dicom"]:
                    if not self._include_videos:
                        continue
                    for frame_num, frame_item in data_unit["labels"].items():
                        image_id = self.get_image_id(data_hash, int(frame_num))
                        objects = frame_item["objects"]
                        object_metrics = copy.deepcopy(data_unit_metrics[f"{frame_num:05d}"])
                        annotations.extend(self.get_annotation(objects, object_metrics, image_id))

                else:
                    image_id = self.get_image_id(data_hash)
                    objects = data_unit["labels"]["objects"]
                    frame_num = int(data_unit["data_sequence"])
                    object_metrics = copy.deepcopy(data_unit_metrics[f"{frame_num:05d}"])
                    annotations.extend(self.get_annotation(objects, object_metrics, image_id))

        return annotations

    # DENIS: naming with plural/singular
    def get_annotation(self, objects: List[dict], metrics: dict, image_id: int) -> List[dict]:
        annotations = []
        frame_level = metrics.pop("frame-level")
        for object_ in objects:
            shape = object_["shape"]

            # DENIS: abstract this
            for image_data in self._coco_json["images"]:
                if image_data["id"] == image_id:
                    size = Size(width=image_data["width"], height=image_data["height"])

            # DENIS: would be nice if this shape was an enum => with the Json support.
            if shape == Shape.BOUNDING_BOX.value:
                # DENIS: how can I make sure this can be extended properly? At what point do I transform this to a JSON?
                # maybe I can have an `asdict` if this is a dataclass, else just keep the json and have the return type
                # be a union?!
                res = self.get_bounding_box(object_, image_id, size)
            elif shape == Shape.ROTATABLE_BOUNDING_BOX.value:
                res = self.get_rotatable_bounding_box(object_, image_id, size)
            elif shape == Shape.POLYGON.value:
                res = self.get_polygon(object_, image_id, size)
            elif shape == Shape.POLYLINE.value:
                res = self.get_polyline(object_, image_id, size)
            elif shape == Shape.POINT.value:
                res = self.get_point(object_, image_id, size)
            elif shape == Shape.SKELETON.value:
                res = self.get_skeleton(object_, image_id, size)
            else:
                raise ValueError(f"Unsupported shape: {shape}")

            res_dict = asdict(res)
            res_dict["frame_metrics"] = frame_level
            res_dict["object_metrics"] = metrics.get(object_["objectHash"], {})
            annotations.append(to_attributes_field(res_dict, include_null_annotations=self._include_null_annotations))
        return annotations

    def get_bounding_box(self, object_: dict, image_id: int, size: Size) -> Union[CocoAnnotation, SuperClass]:
        x, y = (
            object_["boundingBox"]["x"] * size.width,
            object_["boundingBox"]["y"] * size.height,
        )
        w, h = (
            object_["boundingBox"]["w"] * size.width,
            object_["boundingBox"]["h"] * size.height,
        )
        area = w * h
        segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        bbox = (x, y, w, h)
        category_id = self.get_category_id(object_)
        id_, iscrowd, track_id, encord_track_uuid = self.get_coco_annotation_default_fields(object_)

        return CocoAnnotation(
            area,
            bbox,
            category_id,
            id_,
            image_id,
            iscrowd,
            segmentation,
            track_id=track_id,
            encord_track_uuid=encord_track_uuid,
        )

    def get_rotatable_bounding_box(self, object_: dict, image_id: int, size: Size) -> Union[CocoAnnotation, SuperClass]:
        x, y = (
            object_["rotatableBoundingBox"]["x"] * size.width,
            object_["rotatableBoundingBox"]["y"] * size.height,
        )
        w, h = (
            object_["rotatableBoundingBox"]["w"] * size.width,
            object_["rotatableBoundingBox"]["h"] * size.height,
        )
        area = w * h
        segmentation = [[x, y, x + w, y, x + w, y + h, x, y + h]]
        bbox = (x, y, w, h)
        category_id = self.get_category_id(object_)
        id_, iscrowd, track_id, encord_track_uuid = self.get_coco_annotation_default_fields(object_)
        rotation = object_["rotatableBoundingBox"]["theta"]

        return CocoAnnotation(
            area,
            bbox,
            category_id,
            id_,
            image_id,
            iscrowd,
            segmentation,
            track_id=track_id,
            encord_track_uuid=encord_track_uuid,
            rotation=rotation,
        )

    def get_polygon(self, object_: dict, image_id: int, size: Size) -> Union[CocoAnnotation, SuperClass]:
        polygon = get_polygon_from_dict(object_["polygon"], size.width, size.height)
        segmentation = [list(chain(*polygon))]
        polygon = Polygon(polygon)
        area = polygon.area
        x, y, x_max, y_max = polygon.bounds
        w, h = x_max - x, y_max - y

        bbox = (x, y, w, h)
        category_id = self.get_category_id(object_)
        id_, iscrowd, track_id, encord_track_uuid = self.get_coco_annotation_default_fields(object_)

        return CocoAnnotation(
            area,
            bbox,
            category_id,
            id_,
            image_id,
            iscrowd,
            segmentation,
            track_id=track_id,
            encord_track_uuid=encord_track_uuid,
        )

    def get_polyline(self, object_: dict, image_id: int, size: Size) -> Union[CocoAnnotation, SuperClass]:
        """Polylines are technically not supported in COCO, but here we use a trick to allow a representation."""
        polygon = get_polygon_from_dict(object_["polyline"], size.width, size.height)
        polyline_coordinate = self.join_polyline_from_polygon(list(chain(*polygon)))
        segmentation = [polyline_coordinate]
        area = 0
        bbox = self.get_bbox_for_polyline(polygon)
        category_id = self.get_category_id(object_)
        id_, iscrowd, track_id, encord_track_uuid = self.get_coco_annotation_default_fields(object_)

        return CocoAnnotation(
            area,
            bbox,
            category_id,
            id_,
            image_id,
            iscrowd,
            segmentation,
            track_id=track_id,
            encord_track_uuid=encord_track_uuid,
        )

    def get_bbox_for_polyline(self, polygon: list) -> CocoBbox:
        if len(polygon) == 2:
            # We have the edge case of a single edge polygon.
            first_point = polygon[0]
            second_point = polygon[1]
            x = min(first_point[0], second_point[0])
            y = min(first_point[1], second_point[1])
            w = abs(first_point[0] - second_point[0])
            h = abs(first_point[1] - second_point[1])
        else:
            polygon_shapely = Polygon(polygon)
            x, y, x_max, y_max = polygon_shapely.bounds
            w, h = x_max - x, y_max - y
        return (x, y, w, h)

    @staticmethod
    def join_polyline_from_polygon(polygon: List[float]) -> List[float]:
        """
        Essentially a trick to represent a polyline in coco. We pretend for this to be a polygon and join every
        coordinate from the end back to the beginning, so it will essentially be an area-less polygon.
        This function technically changes the input polygon in place.
        """
        if len(polygon) % 2 != 0:
            raise RuntimeError("The polygon has an unaccepted shape.")

        idx = len(polygon) - 2
        while idx >= 0:
            y_coordinate = polygon[idx]
            x_coordinate = polygon[idx + 1]
            polygon.append(y_coordinate)
            polygon.append(x_coordinate)
            idx -= 2

        return polygon

    def get_point(self, object_: dict, image_id: int, size: Size) -> Union[CocoAnnotation, SuperClass]:
        x, y = (
            object_["point"]["0"]["x"] * size.width,
            object_["point"]["0"]["y"] * size.height,
        )
        w, h = 0, 0
        area = 0
        segmentation = [[x, y]]
        keypoints = [x, y, 2]
        num_keypoints = 1

        bbox = (x, y, w, h)
        category_id = self.get_category_id(object_)
        id_, iscrowd, track_id, encord_track_uuid = self.get_coco_annotation_default_fields(object_)

        return CocoAnnotation(
            area,
            bbox,
            category_id,
            id_,
            image_id,
            iscrowd,
            segmentation,
            keypoints,
            num_keypoints,
            track_id=track_id,
            encord_track_uuid=encord_track_uuid,
        )

    def get_skeleton(self, object_: dict, image_id: int, size: Size) -> Union[CocoAnnotation, SuperClass]:
        # DENIS: next up: check how this is visualised.
        area = 0
        segmentation: List = []
        keypoints: List = []
        for point in object_["skeleton"].values():
            keypoints += [
                point["x"] * size.width,
                point["y"] * size.height,
                2,
            ]
        num_keypoints = len(keypoints) // 3
        xs, ys = (
            keypoints[::3],
            keypoints[1::3],
        )
        x, y, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
        w, h = x_max - x, y_max - y

        # DENIS: think if the next two lines should be in `get_coco_annotation_default_fields`
        bbox = (x, y, w, h)
        category_id = self.get_category_id(object_)
        id_, iscrowd, track_id, encord_track_uuid = self.get_coco_annotation_default_fields(object_)

        return CocoAnnotation(
            area,
            bbox,
            category_id,
            id_,
            image_id,
            iscrowd,
            segmentation,
            keypoints,
            num_keypoints,
            track_id=track_id,
            encord_track_uuid=encord_track_uuid,
        )

    def get_category_id(self, object_: dict) -> int:
        feature_hash = object_["featureHash"]
        try:
            return self._feature_hash_to_coco_category_id_map[feature_hash]
        except KeyError:
            raise EncodingError(
                f"The feature_hash `{feature_hash}` was not found in the provided ontology. Please "
                f"ensure that the ontology matches the labels provided."
            )

    def get_coco_annotation_default_fields(self, object_: dict) -> Tuple[int, int, Optional[int], Optional[str]]:
        id_ = self.next_annotation_id()
        iscrowd = 0
        track_id = self.get_and_set_track_id(object_hash=object_["objectHash"])
        encord_track_uuid = object_["objectHash"]
        return id_, iscrowd, track_id, encord_track_uuid

    def next_annotation_id(self) -> int:
        next_ = self._current_annotation_id
        self._current_annotation_id += 1
        return next_

    def get_and_set_track_id(self, object_hash: str) -> int:
        if object_hash in self._object_hash_to_track_id_map:
            return self._object_hash_to_track_id_map[object_hash]
        else:
            next_track_id = len(self._object_hash_to_track_id_map)
            self._object_hash_to_track_id_map[object_hash] = next_track_id
            return next_track_id

    def download_image(self, url: str, path: Path):
        """Check if directory exists, create the directory if needed, download the file, store it into the path."""
        path.parent.mkdir(parents=True, exist_ok=True)
        download_file(url, path)

    def get_image_id(self, data_hash: str, frame_num: int = 0) -> int:
        return self._data_hash_to_image_id_map[(data_hash, frame_num)]


def download_file(
    url: str,
    destination: Path,
):

    r = requests.get(url, stream=True)
    with open(destination, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()
    return destination


def extract_frames(video_file_name: Path, img_dir: Path, data_hash: str):
    logger.info(f"Extracting frames from video: {video_file_name}")

    # DENIS: for the rest to work, I will need to throw if the current directory exists and give a nice user warning.
    img_dir.mkdir(parents=True, exist_ok=True)
    command = f"ffmpeg -i {video_file_name} -start_number 0 {img_dir}/{data_hash}_%d.png -hide_banner"
    if subprocess.run(command, shell=True, capture_output=True, stdout=None, check=True).returncode != 0:
        raise RuntimeError(
            "Splitting videos into multiple image files failed. Please ensure that you have FFMPEG "
            f"installed on your machine: https://ffmpeg.org/download.html The comamand that failed was `{command}`."
        )


def generate_coco_file(df: pd.DataFrame, project_dir: Path, ontology_file: Path) -> dict:
    """
    Generate coco JSON file given dataframe.

    Args:
        df: dataframe with selected samples

    Returns:
        dict: Dictionary object of COCO annotations
    """
    # Load label rows and get metrics dict
    label_rows = load_label_rows(df, project_dir / "data")
    metrics = df_to_nested_dict(df)

    # Load ontology json to dict
    with open(ontology_file, "r", encoding="utf-8") as f:
        ontology_dict = json.load(f)

    # Generate COCO annotations file
    encoder = CocoEncoder(
        labels_list=list(label_rows.values()),
        metrics=metrics,
        ontology=OntologyStructure.from_dict(ontology_dict),
    )
    coco_json = encoder.encode(
        download_files=True,
        download_file_path=project_dir,
    )

    return coco_json


def df_to_nested_dict(df: pd.DataFrame) -> dict:
    """
    Convert dataframe of metrics to nested dictionary with the following structure:
        "<label_hash>": {
            "<data_hash>": {
                "<frame_number>": {
                    "frame-level": {frame level metrics},
                    "<object_hash_0>": {object level metrics},
                    "<object_hash_1>": {object level metrics},
                    ...
                }
            }
        }
    Args:
        df: dataframe of metrics

    Returns:
        dict: nested dictionary of metrics
    """
    metrics: Dict[str, dict] = {}
    for row in tqdm(df.iterrows(), desc="Parsing metrics", total=len(df)):
        row = row[1]
        out = row[0].split("_")
        if len(out) == 3:
            object_hashes = None
            label_hash, data_hash, frame_number = out
        elif len(out) > 3:
            label_hash, data_hash, frame_number, *object_hashes = out
        else:
            raise ValueError(f"Invalid identifier `{row[0]}`")

        # Create metrics dict (enforce the structure stated in this method's docstring)
        label_dict = metrics.setdefault(label_hash, {})
        data_dict = label_dict.setdefault(data_hash, {})
        frame_dict = data_dict.setdefault(frame_number, {})
        frame_dict.setdefault("frame-level", {})

        tags = [t.name for t in (row["tags"] if "tags" in row and isinstance(row["tags"], list) else [])]
        if object_hashes is None:  # Frame level metric
            frame_dict.setdefault("frame-level", {}).update(
                {k: v for k, v in row.items() if k not in ["identifier", "url", "tags"] and not pd.isnull(v)}
            )
            frame_dict["frame-level"].setdefault("tags", []).extend(tags)
        else:  # Object level metric
            for object_hash in object_hashes:
                frame_dict.setdefault(object_hash, {}).update(
                    {k: v for k, v in row.items() if k not in ["identifier", "url", "tags"] and not pd.isnull(v)}
                )
                frame_dict[object_hash].setdefault("tags", []).extend(tags)
    return metrics


def load_label_rows(df: pd.DataFrame, data_dir: Path) -> dict[str, dict]:
    """
    Given a dataframe with selected samples, load the label row jsons into a dictionary.

    Args:
        df: dataframe with selected samples

    Returns:
        dict: Dictionary of label rows
    """
    label_rows: Dict[str, dict] = {}
    for id_tmp in tqdm(df.identifier, desc="Loading labels"):
        label_hash = id_tmp.split("_")[0]
        if label_hash not in label_rows.keys():
            # Read label json and add to dict
            label_json = data_dir.joinpath(label_hash, "label_row.json")
            with open(label_json, "r", encoding="utf-8") as f:
                label_rows[label_hash] = json.load(f)
    return label_rows
