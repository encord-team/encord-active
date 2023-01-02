import logging

import cv2
import numpy as np
import pandas as pd

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.utils import get_du_size, get_object_coordinates
from encord_active.lib.embeddings.writer import CSVEmbeddingWriter

logger = logging.getLogger(__name__)
HU_FILENAME = "hu_moments-embeddings"


def generate_hu_embeddings(iterator: Iterator):
    with CSVEmbeddingWriter(iterator.cache_dir, iterator, prefix=HU_FILENAME) as writer:
        for data_unit, img_pth in iterator.iterate(desc="Generating HU embeddings"):
            size = get_du_size(data_unit, img_pth)
            if not size:
                continue
            height, width = size

            for obj in data_unit["labels"].get("objects", []):
                if obj["shape"] != "polygon":
                    continue

                points = get_object_coordinates(obj)
                if not points:  # avoid corrupted objects without vertices (empty list - [])
                    continue

                polygon = (np.array(points) * (width, height)).astype(np.int32)
                mask = np.zeros((height, width), dtype="uint8")
                mask = cv2.fillPoly(mask, [polygon], 255)
                moments = cv2.HuMoments(cv2.moments(mask)).flatten()

                writer.write(moments.tolist(), obj)


def get_hu_embeddings(iterator: Iterator, *, force: bool = False) -> pd.DataFrame:
    hu_moments_path = iterator.cache_dir / f"embeddings/{HU_FILENAME}.csv"

    if force:
        logger.info("Regenerating embeddings...")
        generate_hu_embeddings(iterator)
        hu_moments_df = pd.read_csv(hu_moments_path)
        logger.info("Done!")
    else:
        try:
            hu_moments_df = pd.read_csv(hu_moments_path)
        except FileNotFoundError:
            logger.info(f"{hu_moments_path} not found. Generating embeddings...")
            generate_hu_embeddings(iterator)
            hu_moments_df = pd.read_csv(hu_moments_path)
            logger.info("Done!")

    hu_moments_df = hu_moments_df.sort_values(["identifier"], ascending=True).reset_index()
    return hu_moments_df
