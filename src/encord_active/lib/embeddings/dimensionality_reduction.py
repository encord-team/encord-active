import pickle
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import umap
from pandera.typing import DataFrame

from encord_active.lib.embeddings.utils import (
    Embedding2DSchema,
    EmbeddingType,
)
from encord_active.lib.project.project_file_structure import ProjectFileStructure

warnings.filterwarnings("ignore", "n_neighbors is larger than the dataset size", category=UserWarning)
MIN_SAMPLES = 4  # The number 4 is experimentally determined, less than this creates error for UMAP calculation


def generate_2d_embedding_data(embedding_type: EmbeddingType, project_dir: Path):
    """
    This function transforms high dimensional embedding data to 2D and saves it to a file
    """
    pfs = ProjectFileStructure(project_dir)

    collections = load_collections(embedding_type, pfs)
    if not collections:
        return

    embeddings = np.array([collection["embedding"] for collection in collections])
    if embeddings.shape[0] < MIN_SAMPLES:
        return

    reducer = umap.UMAP(random_state=0)
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

    save_embeddings
    target_path = pfs.get_embeddings_file(embedding_type, reduced=True)
    target_path.write_bytes(pickle.dumps(embeddings_2d_collection))


def get_2d_embedding_data(
    project_file_structure: ProjectFileStructure, embedding_type: EmbeddingType
) -> Optional[DataFrame[Embedding2DSchema]]:
    embedding_file_path = project_file_structure.get_embeddings_file(embedding_type, reduced=True)

    if not embedding_file_path.exists():
        return None
    with open(embedding_file_path, "rb") as f:
        cnn_embeddings = pickle.load(f)

    df = pd.DataFrame(cnn_embeddings)
    return DataFrame[Embedding2DSchema](df)
