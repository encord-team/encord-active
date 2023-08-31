import orjson

from encord_active.lib.embeddings.embedding_index import EmbeddingIndex
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.project.project_file_structure import ProjectFileStructure


def project_has_embedding_type(pfs: ProjectFileStructure, et: EmbeddingType):
    if et == EmbeddingType.IMAGE:
        return True

    ontology = orjson.loads(pfs.ontology.read_text())  # pylint: disable=no-member
    if et == EmbeddingType.OBJECT:
        return bool(ontology.get("objects", []))
    if et == EmbeddingType.CLASSIFICATION:
        return bool(ontology.get("classifications", []))


def up(pfs: ProjectFileStructure):
    for embedding_type in [EmbeddingType.IMAGE, EmbeddingType.OBJECT, EmbeddingType.CLASSIFICATION]:
        if project_has_embedding_type(pfs, embedding_type):
            EmbeddingIndex.from_project(pfs, embedding_type)
