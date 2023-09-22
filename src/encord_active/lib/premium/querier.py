import os
import uuid
from pathlib import Path
from typing import Optional, Type, TypeVar

import requests
from pydantic import BaseModel
from requests import ConnectionError

from encord_active.lib.premium.model import (
    CodeOnDataFrameSearchResponse,
    CodeSearchResponse,
    SearchResponse,
    SemanticQuery,
    TextQuery,
)

T = TypeVar("T", bound=BaseModel)


class Querier:
    def __init__(self, db_path: Path, project_hash: uuid.UUID):
        self.db_path = db_path
        self.project_hash = project_hash
        self.api_url = os.getenv("PREMIUM_API_URL", "http://localhost:5051")
        self._premium_available: Optional[bool] = None

    def _search_clip(self, query: SemanticQuery, timeout: Optional[float] = None) -> Optional[dict]:
        endpoint = "search/semantic"
        params = {"project_hash": self.project_hash, "db": self.db_path}

        files = []
        if query.image:
            files.append(("image", query.image))

        data = query.dict()
        data.pop("image", None)
        ids = set(data.pop("identifiers", []))

        response = requests.post(f"{self.api_url}/{endpoint}", params=params, data=data, files=files, timeout=timeout)  # type: ignore
        if response.status_code != 200:
            return None

        result = response.json()
        if len(ids):
            result["result_identifiers"] = list(
                filter(lambda id_value: id_value["identifier"] in ids, result["result_identifiers"])
            )

        return result

    def post_data(
        self, endpoint: str = "", data: Optional[dict] = None, timeout: Optional[float] = None
    ) -> Optional[dict]:
        params = {"project": self.project_hash, "db": self.db_path}
        response = requests.post(f"{self.api_url}/{endpoint}", params=params, data=data, timeout=timeout)  # type: ignore
        if response.status_code != 200:
            return None
        return response.json()

    @property
    def premium_available(self) -> bool:
        if self._premium_available is None:
            try:
                self._premium_available = bool(
                    self.post_data(timeout=float(os.getenv("PREMIUM_API_PING_TIMEOUT", 0.5)))
                )
            except (ConnectionError, ConnectionRefusedError, Exception):
                pass
        return self._premium_available or False

    def _search_with(self, endpoint: str, query: BaseModel, response_type: Type[T]) -> Optional[T]:
        res = self.post_data(endpoint, data=query.dict())
        if res is None:
            return res
        return response_type.parse_obj(res)

    def search_semantics(self, query: SemanticQuery) -> Optional[SearchResponse]:
        res = self._search_clip(query)
        if res is None:
            __import__("pdb").set_trace()
            return res
        return SearchResponse.parse_obj(res)

    def search_with_code(self, query: TextQuery) -> Optional[CodeSearchResponse]:
        return self._search_with("search/code", query, CodeSearchResponse)

    def search_with_code_on_dataframe(self, query: TextQuery) -> Optional[CodeOnDataFrameSearchResponse]:
        return self._search_with("search/code_on_dataframe", query, CodeOnDataFrameSearchResponse)
