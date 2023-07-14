from dataclasses import dataclass
from typing import Dict


@dataclass
class EnumDefinition:
    pass


DataEnums: Dict[str, EnumDefinition] = {}
AnnotationEnums: Dict[str, EnumDefinition] = {}
