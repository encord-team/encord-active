import json
import uuid
from typing import List, Optional, Tuple, Union

import numpy as np
import umap

from encord_active.analysis.reductions.umap_reduce import (
    UMAPSerialized,
    deserialize_umap,
    serialize_umap,
)
from encord_active.db.models import EmbeddingReductionType, ProjectEmbeddingReduction

ReductionType = Union[umap.UMAP]


def serialize_reduction(
    reduction: ReductionType,
    name: str,
    description: str,
    project_hash: uuid.UUID,
    reduction_hash: Optional[uuid.UUID] = None,
) -> ProjectEmbeddingReduction:
    if isinstance(reduction, umap.UMAP):
        reduction_type = EmbeddingReductionType.UMAP
        try:
            reduction_bytes = (
                serialize_umap(reduction)
                .json(
                    exclude_defaults=True,
                    indent=0,
                    allow_nan=True,
                    separators=(",", ":"),
                )
                .encode("utf-8")
            )
        except Exception as e:
            print(f"Failed to serialize UMAP reduction: {e}")
            reduction_bytes = b""
    else:
        raise ValueError(f"Unknown project reduction type: {type(reduction)}")
    return ProjectEmbeddingReduction(
        reduction_hash=reduction_hash or uuid.uuid4(),
        reduction_name=name,
        reduction_description=description,
        project_hash=project_hash,
        reduction_type=reduction_type,
        reduction_bytes=reduction_bytes,
    )


def deserialize_reduction(reduction: ProjectEmbeddingReduction) -> ReductionType:
    if reduction.reduction_type == EmbeddingReductionType.UMAP:
        return deserialize_umap(UMAPSerialized.parse_obj(json.loads(reduction.reduction_bytes.decode("utf-8"))))
    else:
        raise ValueError(f"Unknown project reduction type: {reduction.reduction_type}")


def create_reduction(
    reduction_type: EmbeddingReductionType,
    train_samples: List[np.ndarray],
) -> ReductionType:
    if reduction_type == EmbeddingReductionType.UMAP:
        umap_res = umap.UMAP(random_state=0)
        umap_res.fit(np.stack(train_samples))
        return umap_res
    else:
        raise ValueError(f"Unknown project reduction type: {reduction_type}")


def apply_embedding_reduction(embeddings: List[np.ndarray], reduction: ReductionType) -> List[Tuple[float, float]]:
    if len(embeddings) == 0:
        return []
    if isinstance(reduction, umap.UMAP):
        transformed = reduction.transform(np.stack(embeddings))
        return [(float(x), float(y)) for x, y in transformed]
    else:
        raise ValueError(f"Unknown project reduction type: {type(reduction)}")
