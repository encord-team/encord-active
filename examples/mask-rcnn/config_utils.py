import configparser
import pathlib
from pathlib import Path
from types import SimpleNamespace
from typing import List, Tuple


def get_config(path: str):
    """Read config from file.
    :param path: path to config file.
    :return: nested SimpleNamespace object with each section of the config.ini.
    """
    config = configparser.ConfigParser()
    config.read(path)
    params = {}
    for section in config.sections():
        d = {}
        for key in config[section]:
            # Convert each value to the appropriate type
            try:
                value = eval(config[section][key])
            except:
                value = config[section][key]
            if section == "PATHS":
                value = Path(value).expanduser()
            d[key] = value
        params[section.lower()] = SimpleNamespace(**d)
    return SimpleNamespace(**params)


def args_type(default):
    if isinstance(default, bool):
        return lambda x: bool(["False", "True"].index(x))
    if isinstance(default, int):
        return lambda x: float(x) if ("e" in x or "." in x) else int(x)
    if isinstance(default, pathlib.Path):
        return lambda x: pathlib.Path(x).expanduser()
    return type(default)


def flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
    items: List[Tuple] = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, SimpleNamespace):
            items.extend(flatten_dict(vars(v), new_key, sep=sep).items())
        else:
            items.append((new_key, v))  # type: ignore
    return dict(items)  # type: ignore
