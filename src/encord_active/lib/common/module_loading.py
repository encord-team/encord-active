import importlib.util
from pathlib import Path
from typing import Union


class ModuleLoadError(Exception):
    def __init__(self, e: Exception, module_path: Path) -> None:
        self.module_path = module_path
        self.e = e
        super().__init__(str(self))

    def __str__(self) -> str:
        return f"Failed to load module '{self.module_path}' with error {self.e}"


def load_module(module_path: Union[str, Path]):
    if isinstance(module_path, str):
        module_path = Path(module_path)

    # Load the module from its full path
    if module_path.suffix != ".py":
        raise ValueError(f"Module '{module_path.as_posix()}' doesn't have a valid python module extension (py).")
    try:
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path.as_posix())
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore
    except Exception as e:
        raise ModuleLoadError(e, module_path)
    return mod
