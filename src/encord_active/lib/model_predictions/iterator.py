import json
import logging
from copy import deepcopy
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pytz
from encord.objects.common import Shape
from encord.objects.ontology_object import Object
from pandas import Series
from tqdm.auto import tqdm

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.utils import rle_to_binary_mask

logger = logging.getLogger(__name__)
GMT_TIMEZONE = pytz.timezone("GMT")
DATETIME_STRING_FORMAT = "%a, %d %b %Y %H:%M:%S %Z"
BBOX_KEYS = {"x", "y", "h", "w"}


# === UTILITIES === #
def get_timestamp():
    now = datetime.now()
    new_timezone_timestamp = now.astimezone(GMT_TIMEZONE)
    return new_timezone_timestamp.strftime(DATETIME_STRING_FORMAT)


def lower_snake_case(s: str):
    return "_".join(s.lower().split())


class PredictionIterator(Iterator):
    def __init__(self, cache_dir: Path, subset_size: Optional[int] = None, **kwargs):
        super().__init__(cache_dir, subset_size, **kwargs)
        label_hashes = set(self.label_rows.keys())

        # Predictions
        predictions_file = cache_dir / "predictions" / "predictions.csv"
        predictions = pd.read_csv(predictions_file, index_col=0)
        self.length = predictions["img_id"].nunique()

        identifiers = predictions["identifier"].str.split("_", expand=True)
        identifiers.columns = ["label_hash", "du_hash", "frame", "object_hash"][: len(identifiers.columns)]
        identifiers["frame"] = pd.to_numeric(identifiers["frame"])

        predictions = pd.concat([predictions, identifiers], axis=1)
        predictions["pidx"] = predictions.index

        self.predictions = predictions[predictions["label_hash"].isin(label_hashes)]

        # Class index
        class_idx_file = cache_dir / "predictions" / "class_idx.json"
        with class_idx_file.open("r", encoding="utf-8") as f:
            class_idx = json.load(f)

        self.ontology_objects: Dict[int, Object] = {
            int(k): next(o for o in self.project.ontology.objects if v["featureHash"] == o.feature_node_hash)
            for k, v in class_idx.items()
        }
        self.row_cache: List[Tuple[str, str, int, Dict[Any, Any], Optional[Path]]] = []

    def get_image_path(self, pred: Series) -> Optional[Path]:
        images_dir = self.project.file_structure.label_row_structure(pred["label_hash"]).images_dir
        du_hash = pred["du_hash"]
        image_options = list(images_dir.glob(f"{du_hash}.*"))
        if len(image_options) == 1:
            return image_options[0]
        elif len(image_options) > 1:
            re_matches = [frame_file for frame_file in image_options if frame_file.stem == f"{du_hash}_{pred['frame']}"]
            if re_matches:
                return re_matches[0]
        return None

    def get_encord_object(self, pred: Tuple[Any, Series], width: int, height: int):
        _pred = pred[1]
        ontology_object: Object = self.ontology_objects[_pred["class_id"]]
        if ontology_object.shape == Shape.BOUNDING_BOX:
            x1, y1, x2, y2 = _pred["x1"], _pred["y1"], _pred["x2"], _pred["y2"]
            object_data = {
                "x": x1 / width,
                "y": y1 / height,
                "w": (x2 - x1) / width,
                "h": (y2 - y1) / height,
            }
        else:
            mask = rle_to_binary_mask(eval(_pred["rle"]))
            # Note: This approach may generate invalid polygons (self crossing) :-(
            contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            if len(contours) > 1:
                max_idx = np.argmax(list(map(cv2.contourArea, contours)))
                contour = contours[max_idx].reshape((-1, 2))
            else:
                contour = contours[0].reshape((-1, 2))

            if contour.shape[0] < 3:
                pass
                # logger.warning("Skipping contour with less than 3 vertices.")
            object_data = {
                str(i): {"x": round(c[0] / width, 4), "y": round(c[1] / height, 4)} for i, c in enumerate(contour)
            }

        object_hash = _pred["identifier"].rsplit("_", 1)[1]
        timestamp: str = get_timestamp()
        shape: str = ontology_object.shape.value

        object_dict = {
            "name": ontology_object.name,
            "color": ontology_object.color,
            "value": lower_snake_case(ontology_object.name),
            "createdAt": timestamp,
            "createdBy": "model_predictions@encord.com",
            "confidence": _pred["confidence"],
            "objectHash": object_hash,
            "featureHash": ontology_object.feature_node_hash,
            "lastEditedAt": timestamp,
            "lastEditedBy": "model_predictions@encord.com",
            "shape": shape,
            "manualAnnotation": False,
            "reviews": [],
        }

        if shape == "bounding_box":
            object_dict["boundingBox"] = {k: round(v, 4) for k, v in object_data.items()}
        elif shape == "polygon":
            object_dict["polygon"] = object_data

        return object_dict

    def iterate(self, desc: str = "") -> Generator[Tuple[dict, Optional[Path]], None, None]:
        pbar = tqdm(total=self.length, desc=desc, leave=False)
        if self.row_cache:
            for self.label_hash, self.du_hash, self.frame, du, pth in self.row_cache:
                yield du, pth
                pbar.update(1)
            return

        for label_hash, lh_group in self.predictions.groupby("label_hash"):
            if label_hash not in self.label_rows:
                continue

            self.label_hash = label_hash
            label_row = self.label_rows[label_hash]
            self.dataset_title = label_row["dataset_title"]

            for frame, fr_preds in lh_group.groupby("frame"):
                self.du_hash = fr_preds.iloc[0]["du_hash"]
                self.frame = frame

                du = deepcopy(label_row["data_units"][self.du_hash])
                width = int(du["width"])
                height = int(du["height"])

                du["labels"] = {
                    "objects": list(
                        map(partial(self.get_encord_object, width=width, height=height), fr_preds.iterrows())
                    ),
                    "classifications": [],
                }
                pth = self.get_image_path(fr_preds.iloc[0])
                yield du, pth
                self.row_cache.append((self.label_hash, self.du_hash, self.frame, du, pth))
                pbar.update(1)

    def __len__(self):
        return self.length

    def get_identifier(self, object: Union[dict, list[dict], None] = None, frame: Optional[int] = None) -> Any:
        """
        Note that this only makes sense for scoring each object individually.
        """
        label_hash = self.label_hash
        du_hash = self.du_hash
        frame = self.frame if frame is None else frame

        key = f"{label_hash}_{du_hash}_{frame:05d}"
        if object:
            objects = [object] if isinstance(object, dict) else object
            hashes = [obj["objectHash"] if "objectHash" in obj else obj["featureHash"] for obj in objects]
            return "_".join([key] + hashes)
        return key

    def get_data_url(self):
        # No need for the url as it is in the predictions.csv file already
        return ""

    def get_label_logs(self, object_hash: Optional[str] = None, refresh: bool = False) -> List[dict]:
        # Fail safe
        raise NotImplementedError("Label logs are not available for predictions.")

    @staticmethod
    def update_cache_dir(cache_dir: Path) -> Path:
        # Store prediction specific scores in the predictions subdirectory
        return cache_dir / "predictions"
