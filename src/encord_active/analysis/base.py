from abc import ABC
from typing import Set


class BaseAnalysis(ABC):
    def __init__(self, ident: str, dependencies: Set[str]) -> None:
        self.ident = ident
        self.dependencies = dependencies
