import json
import logging
import typing
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import pytz
from encord.objects.common import Shape
from encord.objects.ontology_object import Object
from pandas import Series
from PIL import Image
from tqdm.auto import tqdm

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.time import get_timestamp
from encord_active.lib.common.utils import rle_to_binary_mask
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.db.predictions import FrameClassification
from encord_active.lib.encord.utils import lower_snake_case
from encord_active.lib.labels.classification import LabelClassification
from encord_active.lib.labels.object import ObjectShape
from encord_active.lib.model_predictions.writer import (
    ClassID,
    ClassificationAttributeOption,
    iterate_classification_attribute_options,
)

if typing.TYPE_CHECKING:
    import prisma

logger = logging.getLogger(__name__)
GMT_TIMEZONE = pytz.timezone("GMT")
DATETIME_STRING_FORMAT = "%a, %d %b %Y %H:%M:%S %Z"
BBOX_KEYS = {"x", "y", "h", "w"}


# === UTILITIES === #
class PredictionIterator(Iterator):
    def __init__(self, cache_dir: Path, subset_size: Optional[int] = None, **kwargs):
        super().__init__(cache_dir, subset_size, **kwargs)
        label_hashes = set(self.label_rows.keys())

        # Predictions
        predictions_file = cache_dir / "predictions" / kwargs["prediction_type"].value / "predictions.csv"
        predictions = pd.read_csv(predictions_file, index_col=0)
        self.length = predictions["img_id"].nunique()

        identifiers = predictions["identifier"].str.split("_", expand=True)
        identifiers.columns = ["label_hash", "du_hash", "frame", "object_hash"][: len(identifiers.columns)]  # type: ignore
        identifiers["frame"] = pd.to_numeric(identifiers["frame"])

        predictions = pd.concat([predictions, identifiers], axis=1)
        predictions["pidx"] = predictions.index

        self.predictions = predictions[predictions["label_hash"].isin(label_hashes)]

        # Class index
        class_idx_file = cache_dir / "predictions" / kwargs["prediction_type"].value / "class_idx.json"
        with class_idx_file.open("r", encoding="utf-8") as f:
            class_idx: Dict[ClassID, dict] = {ClassID(k): v for k, v in json.load(f).items()}

        object_lookup = {obj.feature_node_hash: obj for obj in self.project.ontology.objects}
        classification_lookup = {
            hashes: items for hashes, items in iterate_classification_attribute_options(self.project.ontology)
        }

        self.ontology_objects: Dict[ClassID, Object] = {}
        self.ontology_classifications: Dict[ClassID, ClassificationAttributeOption] = {}
        for class_id, label in class_idx.items():
            if label["featureHash"] in object_lookup:
                self.ontology_objects[class_id] = object_lookup[label["featureHash"]]
            else:
                key = FrameClassification(
                    feature_hash=label["featureHash"],
                    attribute_hash=label["attributeHash"],
                    option_hash=label["optionHash"],
                )
                self.ontology_classifications[class_id] = classification_lookup[key]

    def get_image(self, pred: Series, cache_db: "prisma.Prisma") -> Optional[Image.Image]:
        label_row_structure = self.project.file_structure.label_row_structure(pred["label_hash"])
        frame = pred.to_dict().get("frame", None)
        if frame is not None:
            frame = int(frame)
        data_unit, image = next(
            label_row_structure.iter_data_unit_with_image(
                data_unit_hash=pred["du_hash"], frame=frame, cache_db=cache_db
            )
        )
        return image

    def get_encord_object(self, pred: Series, width: int, height: int, ontology_object: Object):
        if ontology_object.shape == Shape.BOUNDING_BOX:
            x1, y1, x2, y2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]
            object_data = {
                "x": x1 / width,
                "y": y1 / height,
                "w": (x2 - x1) / width,
                "h": (y2 - y1) / height,
            }
        else:
            mask = rle_to_binary_mask(eval(pred["rle"]))
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

        object_hash = pred["identifier"].rsplit("_", 1)[1]
        timestamp: str = get_timestamp()
        shape: str = ontology_object.shape.value

        object_dict = {
            "name": ontology_object.name,
            "color": ontology_object.color,
            "value": lower_snake_case(ontology_object.name),
            "createdAt": timestamp,
            "createdBy": "model_predictions@encord.com",
            "confidence": pred["confidence"],
            "objectHash": object_hash,
            "featureHash": ontology_object.feature_node_hash,
            "lastEditedAt": timestamp,
            "lastEditedBy": "model_predictions@encord.com",
            "shape": shape,
            "manualAnnotation": False,
            "reviews": [],
        }

        if shape == ObjectShape.BOUNDING_BOX:
            object_dict["boundingBox"] = {k: round(v, 4) for k, v in object_data.items()}
        elif shape == ObjectShape.ROTATABLE_BOUNDING_BOX:
            box = {k: round(v, 4) for k, v in object_data.items()}
            object_dict["rotatableBoundingBox"] = {**box, "theta": object_data["theta"]}
        elif shape == ObjectShape.POLYGON:
            object_dict["polygon"] = object_data

        return object_dict

    def get_encord_classification(self, pred: Series, ontology_classification: ClassificationAttributeOption):
        classification_hash = pred["identifier"].rsplit("_", 1)[1]

        return LabelClassification(
            name=ontology_classification.attribute.name,
            value=ontology_classification.attribute.name.lower(),
            reviews=[],
            createdAt=get_timestamp(),
            createdBy="model_predictions@encord.com",
            confidence=pred["confidence"],
            featureHash=ontology_classification.classification.feature_node_hash,
            classificationHash=classification_hash,
            manualAnnotation=False,
        )

    def iterate(self, desc: str = "") -> Generator[Tuple[dict, Optional[Image.Image]], None, None]:
        pbar = tqdm(total=self.length, desc=desc, leave=False)
        with PrismaConnection(self.project_file_structure) as cache_db:
            for label_hash, lh_group in self.predictions.groupby("label_hash"):
                if label_hash not in self.label_rows:
                    continue

                self.label_hash = str(label_hash)
                label_row = self.label_rows[self.label_hash]
                self.dataset_title = label_row["dataset_title"]

                for frame, fr_preds in lh_group.groupby("frame"):
                    self.du_hash = fr_preds.iloc[0]["du_hash"]
                    self.frame = int(str(frame))

                    du = deepcopy(label_row["data_units"][self.du_hash])
                    width = int(du["width"])
                    height = int(du["height"])

                    objects = []
                    classifications = []

                    for _, prediction in fr_preds.iterrows():
                        class_id = prediction.class_id
                        if class_id in self.ontology_objects:
                            objects.append(
                                self.get_encord_object(prediction, width, height, self.ontology_objects[class_id])
                            )
                        elif class_id in self.ontology_classifications:
                            classifications.append(
                                asdict(
                                    self.get_encord_classification(prediction, self.ontology_classifications[class_id])
                                )
                            )
                        else:
                            logger.error("The prediction is not in the ontology objects or classifications")

                    du["labels"] = {"objects": objects, "classifications": classifications}
                    image = self.get_image(fr_preds.iloc[0], cache_db=cache_db)
                    if image is None:
                        logger.error(f"Failed to open Image at frame: {self.du_hash}/{fr_preds.iloc[0]}")
                    yield du, image
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
