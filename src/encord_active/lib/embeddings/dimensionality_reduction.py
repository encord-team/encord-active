import pickle
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from numba.core.errors import NumbaDeprecationWarning
from pandera.typing import DataFrame

from encord_active.lib.embeddings.types import Embedding2DSchema, LabelEmbedding
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.project.project_file_structure import ProjectFileStructure

warnings.filterwarnings("ignore", category=NumbaDeprecationWarning)
warnings.filterwarnings("ignore", "n_neighbors is larger than the dataset size", category=UserWarning)

import umap

MIN_SAMPLES = 4  # The number 4 is experimentally determined, less than this creates error for UMAP calculation


def generate_2d_embedding_data(
    embedding_type: EmbeddingType, project_file_structure: ProjectFileStructure, label_embeddings: list[LabelEmbedding]
):
    """
    This function transforms high dimensional embedding data to 2D and saves it to a file
    """
    if not label_embeddings:
        return

    embeddings = np.array([emb["embedding"] for emb in label_embeddings])
    if embeddings.shape[0] < MIN_SAMPLES:
        return

    reducer = umap.UMAP(random_state=0)
    embeddings_2d = reducer.fit_transform(embeddings)

    embeddings_2d_collection: dict[str, list] = {"identifier": [], "x": [], "y": [], "label": []}
    for label_embedding, emb_2d in zip(label_embeddings, embeddings_2d):
        identifier_no_label = (
            f'{label_embedding["label_row"]}_{label_embedding["data_unit"]}_{int(label_embedding["frame"]):05d}'
        )
        if embedding_type == EmbeddingType.IMAGE:
            embeddings_2d_collection["identifier"].append(identifier_no_label)
            embeddings_2d_collection["label"].append("No label")
        elif embedding_type == EmbeddingType.OBJECT:
            embeddings_2d_collection["identifier"].append(f'{identifier_no_label}_{label_embedding["labelHash"]}')
            embeddings_2d_collection["label"].append(label_embedding["name"])
        elif embedding_type == EmbeddingType.CLASSIFICATION:
            # Due to the following line, currently there is only one classification answer
            # https://github.com/encord-team/encord-active/blob/2e09cedf1c07eb89c91cad928113b1b51fc8dc7f/src/encord_active/lib/embeddings/cnn.py#L238
            embeddings_2d_collection["identifier"].append(f'{identifier_no_label}_{label_embedding["labelHash"]}')
            embeddings_2d_collection["label"].append(
                label_embedding["classification_answers"]["answer_name"]
                if label_embedding["classification_answers"] is not None
                else "No label"
            )

        x, y = emb_2d
        embeddings_2d_collection["x"].append(x)
        embeddings_2d_collection["y"].append(y)

    target_path = project_file_structure.get_embeddings_file(embedding_type, reduced=True)
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
