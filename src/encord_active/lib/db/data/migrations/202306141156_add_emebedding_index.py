from encord_active.lib.embeddings.embedding_index import EmbeddingIndex
from encord_active.lib.metrics.types import EmbeddingType
from encord_active.lib.project.project_file_structure import ProjectFileStructure


def up(pfs: ProjectFileStructure):
    for embedding_type in [EmbeddingType.IMAGE, EmbeddingType.OBJECT, EmbeddingType.CLASSIFICATION]:
        if not EmbeddingIndex.index_available(pfs, embedding_type):
            EmbeddingIndex.from_project(pfs, embedding_type)
