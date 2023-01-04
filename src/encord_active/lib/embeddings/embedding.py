from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from faiss import IndexFlatL2

from encord_active.lib.embeddings.utils import LabelEmbedding
from encord_active.lib.metrics.metric import EmbeddingType


@dataclass
class EmbeddingInformation:
    type: EmbeddingType
    has_annotations: bool = False
    collections: List[LabelEmbedding] = field(default_factory=list)
    question_hash_to_collection_indexes: Dict[str, Any] = field(default_factory=dict)
    keys_having_similarity: Dict[str, Any] = field(default_factory=dict)
    faiss_index_mapping: Optional[Dict[str, IndexFlatL2]] = None
    faiss_index: Optional[IndexFlatL2] = None
    similarities: Dict[str, Union[List[Dict], Dict[str, List[Dict]]]] = field(default_factory=dict)
