import importlib.util
from pathlib import Path
from typing import Union

from loguru import logger


def load_module(module_path: Union[str, Path]):
    if isinstance(module_path, str):
        module_path = Path(module_path)

    # Load the module from its full path
    if module_path.suffix != ".py":
        logger.error(f"Module '{module_path.as_posix()}' doesn't have a valid python module extension (py).")
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path.as_posix())
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore
    except Exception as e:
        logger.error(f"Module '{module_path.as_posix()}' is ill-formed. Exception: {e}")
        return None
    return mod
