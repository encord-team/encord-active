import json
import logging
from typing import List, Dict

import cv2
import numpy as np
import pandas as pd
from prisma import Base64
from prisma.models import EmbeddingRow

from encord_active.lib.common.iterator import Iterator
from encord_active.lib.common.utils import get_du_size, get_object_coordinates
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.embeddings.writer import DBEmbeddingWriter

logger = logging.getLogger(__name__)
HU_FILENAME = "hu_moments-embeddings"


def generate_hu_embeddings(iterator: Iterator):
    with DBEmbeddingWriter(iterator.project_file_structure, iterator, prefix=HU_FILENAME) as writer:
        for data_unit, image in iterator.iterate(desc="Generating HU embeddings"):
            size = get_du_size(data_unit, image)
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


def read_hu_embeddings(iterator: Iterator) -> List[dict]:
    with PrismaConnection(iterator.project_file_structure) as conn:
        rows = conn.embeddingrow.find_many(
            where={
                "metric_prefix": HU_FILENAME,
            }
        )

        def decode_embedding(r) -> dict:
            d = dict(r)
            d["embedding"] = json.loads(Base64.decode(d["embedding"]))
            return d

        return [
            decode_embedding(r) for r in rows
        ]


def get_hu_embeddings_rows(iterator: Iterator, *, force: bool = False) -> List[dict]:
    hu_moments_path = iterator.cache_dir / f"embeddings/{HU_FILENAME}.csv"
    # FIXME: run re-calculation of all embeddings

    if force:
        logger.info("Regenerating embeddings...")
        generate_hu_embeddings(iterator)
        hu_moments_rows = read_hu_embeddings(iterator)
        logger.info("Done!")
    else:
        hu_moments_rows = read_hu_embeddings(iterator)
        if len(hu_moments_rows) == 0:
            logger.info(f"{hu_moments_path} not found. Generating embeddings...")
            generate_hu_embeddings(iterator)
        hu_moments_rows = read_hu_embeddings(iterator)
        logger.info("Done!")
    return hu_moments_rows


def get_hu_embeddings_lookup(iterator: Iterator, force: bool = False) -> Dict[str, List[float]]:
    rows = get_hu_embeddings_rows(iterator, force=force)
    return {
        row["identifier"]: row["embedding"]
        for row in rows
    }


def get_hu_embeddings(iterator: Iterator, *, force: bool = False) -> pd.DataFrame:
    hu_moments_rows = get_hu_embeddings_rows(iterator, force=force)
    hu_moments_df = pd.DataFrame.from_records(hu_moments_rows)
    hu_moments_df = hu_moments_df.sort_values(["identifier"], ascending=True).reset_index()
    return hu_moments_df
