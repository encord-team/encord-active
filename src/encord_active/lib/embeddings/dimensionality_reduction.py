import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import umap
from pandera.typing import DataFrame

from encord_active.lib.embeddings.utils import (
    EMBEDDING_REDUCED_TO_FILENAME,
    Embedding2DSchema,
    EmbeddingType,
    load_collections,
)
from encord_active.lib.metrics.utils import MetricScope


def generate_2d_embedding_data(embedding_type: EmbeddingType, project_dir: Path):
    """
    This function transforms high dimensional embedding data to 2D
    """

    collections = load_collections(embedding_type, project_dir / "embeddings")
    embeddings = np.array([collection["embedding"] for collection in collections])

    reducer = umap.UMAP(random_state=0)
    embeddings_2d = reducer.fit_transform(embeddings)

    embeddings_2d_collection = {"identifier": [], "x": [], "y": []}
    for counter, collection in enumerate(collections):
        embeddings_2d_collection["identifier"].append(
            f'{collection["label_row"]}_{collection["data_unit"]}_{collection["frame"]:05d}'
        )
        embeddings_2d_collection["x"].append(embeddings_2d[counter, 0]),
        embeddings_2d_collection["y"].append(embeddings_2d[counter, 1]),

    target_path = Path(project_dir / "embeddings" / EMBEDDING_REDUCED_TO_FILENAME[EmbeddingType.IMAGE])
    target_path.write_bytes(pickle.dumps(embeddings_2d_collection))


def get_2d_embedding_data(embeddings_path: Path, metric_scope: MetricScope) -> Optional[DataFrame[Embedding2DSchema]]:
    if metric_scope == MetricScope.DATA_QUALITY:
        embedding_file_path = embeddings_path / EMBEDDING_REDUCED_TO_FILENAME[EmbeddingType.IMAGE]
        if not embedding_file_path.exists():
            return None
        with open(embedding_file_path, "rb") as f:
            cnn_embeddings = pickle.load(f)

        df = pd.DataFrame(cnn_embeddings)
        df = DataFrame[Embedding2DSchema](df)

        return df
