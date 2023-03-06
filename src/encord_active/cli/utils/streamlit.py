import sys
from os import environ
from pathlib import Path

from streamlit.web import cli as stcli

from encord_active.app.app_config import app_config
from encord_active.cli.utils.decorators import find_child_projects, is_project
from encord_active.lib.versioning.git import GitVersioner


def prevent_detached_versions(path: Path):
    paths = [path] if is_project(path) else find_child_projects(path)
    for path in paths:
        GitVersioner(path).jump_to("latest")


def launch_streamlit_app(target: Path):
    streamlit_page = (Path(__file__).parents[2] / "app" / "streamlit_entrypoint.py").expanduser().absolute()
    data_dir = target.expanduser().absolute().as_posix()
    sys.argv = ["streamlit", "run", streamlit_page.as_posix(), data_dir]

    # NOTE: we need to set PYTHONPATH for file watching
    environ["PYTHONPATH"] = (Path(__file__).parents[2]).as_posix()

    if app_config.is_dev:
        sys.argv.insert(3, "--server.fileWatcherType")
        sys.argv.insert(4, "auto")

    prevent_detached_versions(target)
    sys.exit(stcli.main())  # pylint: disable=no-value-for-parameter
