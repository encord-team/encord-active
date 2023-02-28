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


def generate_2d_embedding_data(embedding_type: EmbeddingType, project_dir: Path):
    """
    This function transforms high dimensional embedding data to 2D and saves it to a file
    """

    collections = load_collections(embedding_type, project_dir / "embeddings")
    if not collections:
        return

    embeddings = np.array([collection["embedding"] for collection in collections])

    reducer = umap.UMAP(random_state=0, verbose=True)
    embeddings_2d = reducer.fit_transform(embeddings)

    embeddings_2d_collection: dict[str, list] = {"identifier": [], "x": [], "y": [], "label": []}
    for counter, collection in enumerate(collections):
        if embedding_type == EmbeddingType.IMAGE:
            embeddings_2d_collection["identifier"].append(
                f'{collection["label_row"]}_{collection["data_unit"]}_{collection["frame"]:05d}'
            )
            embeddings_2d_collection["label"].append("No label")
        elif embedding_type == EmbeddingType.OBJECT:
            embeddings_2d_collection["identifier"].append(
                f'{collection["label_row"]}_{collection["data_unit"]}_{collection["frame"]:05d}_{collection["labelHash"]}'
            )
            embeddings_2d_collection["label"].append(collection["name"])
        elif embedding_type == EmbeddingType.CLASSIFICATION:
            # Due to the following line, currently there is only one classification answer
            # https://github.com/encord-team/encord-active/blob/2e09cedf1c07eb89c91cad928113b1b51fc8dc7f/src/encord_active/lib/embeddings/cnn.py#L238
            embeddings_2d_collection["identifier"].append(
                f'{collection["label_row"]}_{collection["data_unit"]}_{int(collection["frame"]):05d}_{collection["labelHash"]}'
            )
            embeddings_2d_collection["label"].append(
                collection["classification_answers"]["answer_name"]
                if collection["classification_answers"] is not None
                else "No label"
            )

        embeddings_2d_collection["x"].append(embeddings_2d[counter, 0])
        embeddings_2d_collection["y"].append(embeddings_2d[counter, 1])

    target_path = Path(project_dir / "embeddings" / EMBEDDING_REDUCED_TO_FILENAME[embedding_type])
    target_path.write_bytes(pickle.dumps(embeddings_2d_collection))


def get_2d_embedding_data(
    embeddings_path: Path, embedding_type: EmbeddingType
) -> Optional[DataFrame[Embedding2DSchema]]:

    embedding_file_path = embeddings_path / EMBEDDING_REDUCED_TO_FILENAME[embedding_type]

    if not embedding_file_path.exists():
        return None
    with open(embedding_file_path, "rb") as f:
        cnn_embeddings = pickle.load(f)

    df = pd.DataFrame(cnn_embeddings)
    return DataFrame[Embedding2DSchema](df)
