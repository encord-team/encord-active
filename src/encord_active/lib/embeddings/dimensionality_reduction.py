from pathlib import Path

import numpy as np
import umap
from sklearn.preprocessing import StandardScaler

from encord_active.lib.embeddings.utils import (
    EMBEDDING_TYPE_TO_FILENAME,
    EmbeddingType,
    load_collections,
)


def generate_2d_embedding_data(embedding_type: EmbeddingType, project_dir: Path):
    """
    This function transforms high dimensional embedding data to 2D
    """

    collections = load_collections(embedding_type, project_dir / "embeddings")
    embeddings = np.array([collection["embedding"] for collection in collections])
    embeddings = StandardScaler().fit_transform(embeddings)

    reducer = umap.UMAP()
    embeddings_2d = reducer.fit_transform(embeddings)

    embeddings_2d_collection = {}
    for counter, collection in enumerate(collections):
        embeddings_2d_collection = {
            "label_hash": collection["label_hash"],
            "data_hash": collection["data_unit"],
            "x": embeddings_2d[:, 0],
            "y": embeddings_2d[:, 1],
        }
