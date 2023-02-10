import json
import os
import shutil
import warnings
from concurrent.futures import ThreadPoolExecutor as Executor
from concurrent.futures import as_completed
from itertools import product
from pathlib import Path
from typing import (
    Any,
    Collection,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    TypedDict,
    Union,
)

import av
import cv2
import numpy as np
import requests
import yaml
from encord import Project
from loguru import logger
from shapely.errors import ShapelyDeprecationWarning
from shapely.geometry import Polygon
from tqdm.auto import tqdm

from encord_active.lib.encord.utils import get_client

# Silence shapely deprecation warnings from v1.* to v2.0
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
from encord_active.lib.coco.datastructure import CocoBbox


def load_json(json_file: Path) -> Optional[dict]:
    if not json_file.exists():
        return None

    with json_file.open("r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None


class ProjectMeta(TypedDict):
    project_description: str
    project_hash: str
    project_title: str
    ssh_key_path: str


class ProjectNotFound(Exception):
    """Exception raised when a path doesn't contain a valid project.

    Attributes:
        project_dir -- path to a project directory
    """

    def __init__(self, project_dir):
        self.project_dir = project_dir
        super().__init__(f"Couldn't find meta file for project in `{project_dir}`")


def fetch_project_meta(data_dir: Path) -> ProjectMeta:
    meta_file = data_dir / "project_meta.yaml"
    if not meta_file.is_file():
        raise ProjectNotFound(data_dir)

    with meta_file.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_project_info(data_dir: Path) -> Project:
    project_meta = fetch_project_meta(data_dir)
    # == Key file == #
    if "ssh_key_path" not in project_meta:
        raise ValueError("SSH Key path missing in project metadata.")

    private_key_file = Path(project_meta["ssh_key_path"]).expanduser().absolute()
    client = get_client(private_key_file)

    # == Project hash == #
    if "project_hash" not in project_meta:
        raise ValueError("`project_hash` is missing in project metadata.")
    project_hash = project_meta["project_hash"]
    project = client.get_project(project_hash=project_hash)
    return project


def get_du_size(data_unit: dict, img_pth: Optional[Path] = None) -> Optional[Tuple[int, int]]:
    if "width" in data_unit and "height" in data_unit:
        return int(data_unit["height"]), int(data_unit["width"])

    image_corrupted = False
    if img_pth is not None and img_pth.is_file():
        try:
            image = cv2.imread(img_pth.as_posix())
            return image.shape[:2]
        except Exception:
            image_corrupted = True

    return None


def get_object_coordinates(o: dict) -> Optional[list[tuple[Any, Any]]]:
    """
    Convert Encord object dict into list of coordinates.
    :param o: the Encord object dict.
    :return: the list of coordinates.
    """
    if o["shape"] in ["polygon", "polyline", "point", "skeleton"]:
        points_dict = o[o["shape"]]
        points = [(points_dict[str(i)]["x"], points_dict[str(i)]["y"]) for i in range(len(points_dict))]
    elif o["shape"] == "bounding_box":
        bbox = o["boundingBox"]
        points = [
            (bbox["x"], bbox["y"]),
            (bbox["x"] + bbox["w"], bbox["y"]),
            (bbox["x"] + bbox["w"], bbox["y"] + bbox["h"]),
            (bbox["x"], bbox["y"] + bbox["h"]),
        ]
    else:
        logger.warning(f"Unknown shape {o['shape']} in get_object_coordinates function")
        return None

    return points


def get_polygon(o: dict) -> Optional[Polygon]:
    """
    Convert object dict into shapely polygon.
    :param o: the Encord object dict.
    :return: The polygon object.
    """
    if o["shape"] in ["bounding_box", "polygon"]:
        points = get_object_coordinates(o)
    else:
        logger.warning(f"Unknown shape {o['shape']} in get_polygon function")
        return None

    if points is None or len(points) < 3:
        logger.debug("Less than 3 points")
        return None

    polygon = Polygon(points)
    if not polygon.is_simple:
        logger.debug("Not simple")
        return None

    return polygon


def get_geometry_from_encord_object(obj: dict, w: int, h: int) -> Optional[np.ndarray]:
    """
    Convert Encord object dictionary to polygon coordinates used to draw geometries
    with opencv.
    :param obj: the encord object dict
    :param w: the image width
    :param h: the image height
    :return: The polygon coordinates
    """

    polygon = get_polygon(obj)
    if polygon:
        img_size = np.array([[w, h]])
        return (np.array(polygon.exterior.xy).T * img_size).astype(int)
    else:
        return None


def get_iou(p1: Polygon, p2: Polygon):
    """
    Compute IOU between two polygons. If polygons are invalid, 0 will be returned.
    :param p1: polygon 1
    :param p2: polygon 2
    :return: the IOU.
    """
    try:
        intersect = p1.intersection(p2).area
        union = p1.union(p2).area
        return intersect / union
    except:
        return 0


def slice_video_into_frames(
    video_path: Path, out_dir: Path = None, wanted_frames: Collection[int] = None
) -> Tuple[Dict[int, Path], List[int]]:
    frames_dir = out_dir if out_dir else video_path.parent
    frames_dir_existed = frames_dir.exists()

    sliced_frames: Dict[int, Path] = {}
    dropped_frames: List[int] = []

    if frames_dir_existed:
        frames = {
            int(p.stem.rsplit("_", 1)[-1].split(".", 1)[0]): p for p in frames_dir.iterdir() if p.suffix == ".png"
        }
        if frames:
            return frames, []

    try:
        frames_dir.mkdir(parents=True, exist_ok=True)

        video_name = video_path.stem

        with av.open(str(video_path), mode="r") as container:
            for frame in tqdm(
                container.decode(video=0),
                desc="Extracting frames from video",
                leave=True,
                total=container.streams.video[0].frames,
            ):
                frame_num = frame.index

                if wanted_frames is None or frame_num in wanted_frames:

                    if not frame.is_corrupt:
                        frame_name = f"{video_name}_{frame_num}.png"

                        frame_path = Path(frames_dir, frame_name)
                        frame.to_image().save(frame_path)

                        sliced_frames[frame_num] = frame_path
                    else:
                        dropped_frames.append(frame_num)

            return sliced_frames, dropped_frames
    except Exception:
        if not frames_dir_existed:
            shutil.rmtree(frames_dir)
        else:
            for _, frame_path in sliced_frames.items():
                delete_locally_cached_file(frame_path)
        raise


def delete_locally_cached_file(file_path) -> None:
    """
    Deleted cached file from local storage
    :param file_path: str with local file path
    """
    try:
        os.remove(file_path)
    except Exception:
        logger.exception(f"Failed to remove local file [{file_path}]")


def get_bbox_from_encord_label_object(obj: dict, w: int, h: int) -> Optional[tuple]:
    transformed_obj = get_geometry_from_encord_object(obj, w, h)
    if transformed_obj is not None:
        return cv2.boundingRect(transformed_obj)
    else:
        logger.debug("Detected invalid polygon (self-crossing or less than 3 vertices).")
        return None


def fix_duplicate_image_orders_in_knn_graph_all_rows(nearest_items: np.ndarray) -> np.ndarray:
    """
    Duplicate images create problem in nearest neighbor order, for example for index 6 its closest
    neighbors can be [5,6,1,9,3] if 5 and 6 is duplicate, it should be [6,5,1,9,3]. This function ensures that
    the first item is the queried index.

    :param nearest_items: nearest metrics obtained from search method of faiss index
    :return: fixed nearest metrics
    """
    for i, row in enumerate(nearest_items):
        if i != row[0]:
            target_index = np.where(row == i)
            if len(target_index[0]) == 0:
                row[0] = i
            else:
                row[0], row[target_index[0][0]] = row[target_index[0][0]], row[0]

    return nearest_items


def fix_duplicate_image_orders_in_knn_graph_single_row(row_no: int, nearest_items: np.ndarray) -> np.ndarray:
    """
    Duplicate images create problem in nearest neighbor order, for example for index 6 its closest
    neighbors can be [5,6,1,9,3] if 5 and 6 is duplicate, it should be [6,5,1,9,3]. This function ensures that
    the first item is the queried index.

    :param nearest_items: nearest metrics obtained from search method of faiss index
    :return: fixed nearest metrics
    """
    if nearest_items[0, 0] == row_no:
        return nearest_items
    else:
        target_index = np.where(nearest_items[0] == row_no)
        nearest_items[0, 0], nearest_items[0, target_index[0]] = nearest_items[0, target_index[0]], nearest_items[0, 0]
        return nearest_items


class RLEData(TypedDict):
    size: List[int]
    counts: List[int]


def binary_mask_to_rle(mask: np.ndarray) -> RLEData:
    """
    Converts a binary mask into a Run-length Encoding (RLE).

    :param binary_mask: A [h, w] binary mask.
    :return: A dict `{"size": [h, w], "counts": [0, 3, 21, 3, ... ]}`.
    """
    flat_mask = mask.flatten()
    fix_first = flat_mask[0] == 1
    flat_mask = np.pad(flat_mask, (1, 1), "constant", constant_values=(1 - flat_mask[0], 1 - flat_mask[-1]))

    switch_indices = np.where(flat_mask[1:] != flat_mask[:-1])[0]
    rle = switch_indices[1:] - switch_indices[:-1]

    if fix_first:
        rle = np.pad(rle, (1, 0), "constant", constant_values=(0, 0))

    return {"size": list(mask.shape), "counts": rle.tolist()}


def rle_to_binary_mask(rle: RLEData) -> np.ndarray:
    """
    RLE to binary mask.

    :param rle: The run-length encoding dictinary.
    :return: a binary mask of shape `rle["size"]`.
    """
    counts = rle["counts"]
    size = rle["size"]
    mask = np.zeros(np.prod(size), dtype=np.uint8)

    val = 0
    idx = 0
    for count in counts:
        if val == 1:
            mask[idx : idx + count] = val
        val = 1 - val
        idx += count

    return mask.reshape(*size)


def mask_iou(m1: np.ndarray, m2: np.ndarray) -> float:
    intersection = (m1 & m2).sum()  # type: ignore
    union = (m1 | m2).sum()  # type: ignore
    if union == 0:
        return 0.0
    return intersection / union


def __rle_iou(rle1: RLEData, rle2: RLEData) -> float:
    """
    Compute intersection over union for rle dictionaries.
    """
    m1 = rle_to_binary_mask(rle1)
    m2 = rle_to_binary_mask(rle2)
    return mask_iou(m1, m2)


def rle_iou(rle1: Union[RLEData, list[RLEData]], rle2: Union[RLEData, list[RLEData]]) -> np.ndarray:
    """
    Compute intersection over union for rle dictionaries.
    """
    if not isinstance(rle1, list):
        rle1 = [rle1]
    if not isinstance(rle2, list):
        rle2 = [rle2]

    with Executor(max_workers=os.cpu_count()) as exe:
        jobs = [exe.submit(__rle_iou, r1, r2) for r1, r2 in product(rle1, rle2)]
    ious = [job.result() for job in jobs]
    return np.array(ious).reshape(len(rle1), len(rle2))


def mask_to_polygon(mask: np.ndarray) -> Tuple[Optional[List[Any]], CocoBbox]:
    [x, y, w, h] = cv2.boundingRect(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            return contour.squeeze(1).tolist(), (x, y, w, h)

    return None, (x, y, w, h)


def collect_async(fn, job_args, key_fn, max_workers=min(10, (os.cpu_count() or 1) + 4), **kwargs):
    """
    Distribute work across multiple workers. Good for, e.g., downloading data.
    Will return results in dictionary.
    :param fn: The function to be applied
    :param job_args: Arguments to `fn`.
    :param key_fn: Function to determine dictionary key for the result (given the same input as `fn`).
    :param max_workers: Number of workers to distribute work over.
    :param kwargs: Arguments passed on to tqdm.
    :return: Dictionary {key_fn(*job_args): fn(*job_args)}
    """
    job_args = list(job_args)
    if not isinstance(job_args[0], tuple):
        job_args = [(j,) for j in job_args]

    results = {}
    with tqdm(total=len(job_args), **kwargs) as pbar:
        with Executor(max_workers=max_workers) as exe:
            jobs = {exe.submit(fn, *args): key_fn(*args) for args in job_args}
            for job in as_completed(jobs):
                key = jobs[job]

                result = job.result()
                if result is not None:
                    results[key] = result

                pbar.update(1)
    return results


def download_file(
    url: str,
    destination: Path,
    byte_size=1024,
):
    if destination.is_file():
        return destination

    r = requests.get(url, stream=True)

    if r.status_code != 200:
        raise Exception(f"Something happened, couldn't download file from: {url}")

    with destination.open("wb") as f:
        for chunk in r.iter_content(chunk_size=byte_size):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)
                f.flush()

    return destination


def iterate_in_batches(seq: Sequence, size: int):
    return (seq[pos : pos + size] for pos in range(0, len(seq), size))
