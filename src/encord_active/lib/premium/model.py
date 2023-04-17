from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, root_validator


class IdentifierValue(BaseModel):
    identifier: str
    value: Optional[float] = None


class SubsetQueryDefinition(BaseModel):
    identifiers: List[str]


class TextQuery(SubsetQueryDefinition):
    text: str
    limit: Optional[int]


class CLIPQuery(SubsetQueryDefinition):
    text: Optional[str] = None
    image: Optional[Path] = None
    """
    Path to the image
    """

    @root_validator(allow_reuse=True)
    @classmethod
    def validate_at_least_one(cls, values):
        text = values.get("text")
        image = values.get("image")
        assert text is not None or image is not None
        return values


class SearchResponse(BaseModel):
    result_identifiers: List[IdentifierValue]
    is_ordered: bool


class CodeSearchResponse(SearchResponse):
    snippet: str


class EmbeddingQuery(BaseModel):
    text: Optional[str]
    image: Optional[str]
    """
    Path to the image
    """

    @root_validator(allow_reuse=True)
    @classmethod
    def validate_at_least_one(cls, values):
        text = values.get("text")
        image = values.get("image")
        assert text is not None or image is not None
        return values


class EmbeddingResponse(BaseModel):
    embedding: List[float]
