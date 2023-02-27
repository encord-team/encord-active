import importlib.util
import inspect
from pathlib import Path
from typing import Callable, Optional, Union

from encord_active.lib.metrics.execute import logger
from encord_active.lib.metrics.metric import Metric, SimpleMetric


def fill_project_meta_with_builtin_metrics(project_meta: dict):
    project_metrics = project_meta.setdefault("metrics", dict())
    project_metrics.update((metric.__name__, module_path.as_posix()) for metric, module_path in get_builtin_metrics())


def get_builtin_metrics() -> list[tuple[Union[Metric, SimpleMetric], Path]]:
    metrics = []
    module_dirs = ["geometric", "heuristic", "semantic"]
    for dir_name in module_dirs:
        dir_path = Path(__file__).parent / dir_name
        for module_path in dir_path.glob("[!_]*[!_].py"):
            module_metrics = get_module_metrics(module_path, lambda x: True)
            if module_metrics is None:
                continue
            metrics.extend((metric, module_path) for metric in module_metrics)
    return metrics


def get_metrics(
    modules: list[tuple[str, Union[str, Path]]], filter_func=lambda x: True
) -> list[Union[Metric, SimpleMetric]]:
    # can be improved using module.__getattribute__(class_name)
    metrics = []
    for metric_name, module_path in modules:
        module_metrics = get_module_metrics(module_path, lambda x: x.__name__ == metric_name and filter_func(x))
        if module_metrics is None:
            continue
        metrics.extend(module_metrics)
    return metrics


def get_module_metrics(
    module_path: Union[str, Path], filter_func: Callable
) -> Optional[list[Union[Metric, SimpleMetric]]]:
    mod = load_module(module_path)
    if mod is None:
        return None
    cls_members = inspect.getmembers(mod, inspect.isclass)
    metrics = []
    for cls_name, cls_obj in cls_members:
        if (
            (issubclass(cls_obj, Metric) and cls_obj != Metric)
            or (issubclass(cls_obj, SimpleMetric) and cls_obj != SimpleMetric)
        ) and filter_func(cls_obj):
            metrics.append(cls_obj)
    return metrics


def load_module(module_path: Union[str, Path]):
    if isinstance(module_path, str):
        module_path = Path(module_path)

    # Load the module from its full path
    if module_path.suffix != ".py":
        logger.warning(f"Module '{module_path.as_posix()}' doesn't have a valid python module extension (py).")
        return None
    try:
        spec = importlib.util.spec_from_file_location(module_path.stem, module_path.as_posix())
        mod = importlib.util.module_from_spec(spec)  # type: ignore
        spec.loader.exec_module(mod)  # type: ignore
    except Exception as e:
        logger.warning(f"Module '{module_path.as_posix()}' is ill-formed. Exception: {e}")
        return None
    return mod
