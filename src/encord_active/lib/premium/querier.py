import os
from typing import Optional, Type, TypeVar

import requests
from pydantic import BaseModel
from requests import ConnectionError

from encord_active.lib.premium.model import (
    CodeSearchResponse,
    SearchResponse,
    TextQuery,
)
from encord_active.lib.project.project_file_structure import ProjectFileStructure

T = TypeVar("T", bound=BaseModel)


class Querier:
    def __init__(self, pfs: ProjectFileStructure):
        self._pfs = pfs
        self.api_url = os.getenv("PREMIUM_API_URL", "http://localhost:5051")
        self._premium_available: Optional[bool] = None

    def execute(
        self, endpoint: str = "", data: Optional[dict] = None, timeout: Optional[float] = None
    ) -> Optional[dict]:
        params = {"project": self._pfs.project_dir.as_posix()}
        response = requests.post(f"{self.api_url}/{endpoint}", params=params, json=data, timeout=timeout)
        if response.status_code != 200:
            return None

        return response.json()

    @property
    def premium_available(self) -> bool:
        if self._premium_available is None:
            try:
                self._premium_available = bool(self.execute(timeout=float(os.getenv("PREMIUM_API_PING_TIMEOUT", 0.5))))
            except (ConnectionError, ConnectionRefusedError, Exception):
                pass
        return self._premium_available or False

    def _search_with(self, endpoint: str, query: BaseModel, response_type: Type[T]) -> Optional[T]:
        res = self.execute(endpoint, data=query.dict())
        if res is None:
            return res
        else:
            return response_type.parse_obj(res)

    def search_semantics(self, query: TextQuery) -> Optional[SearchResponse]:
        return self._search_with("search/semantic", query, SearchResponse)

    def search_with_code(self, query: TextQuery) -> Optional[CodeSearchResponse]:
        return self._search_with("search/code", query, CodeSearchResponse)
