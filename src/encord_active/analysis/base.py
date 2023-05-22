from abc import ABC
from typing import Set


class BaseEvaluation(ABC):
    def __init__(self, ident: str, dependencies: Set[str]) -> None:
        self.ident = ident
        self.dependencies = dependencies


class BaseAnalysis(BaseEvaluation):
    def __init__(self, ident: str, dependencies: Set[str], long_name: str, short_desc: str, long_desc: str) -> None:
        super().__init__(self, ident, dependencies)
        self.long_name = long_name
        self.short_desc = short_desc
        self.long_desc = long_desc
