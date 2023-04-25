import subprocess
import sys
from os import environ
from pathlib import Path

from streamlit.web import cli as stcli

from encord_active.app.app_config import app_config
from encord_active.cli.utils.decorators import find_child_projects, is_project
from encord_active.lib.db.merged_metrics import ensure_initialised_merged_metrics
from encord_active.lib.versioning.git import GitVersioner


def ensure_safe_project(path: Path):
    paths = [path] if is_project(path) else find_child_projects(path)

    for path in paths:
        GitVersioner(path).jump_to("latest")
        ensure_initialised_merged_metrics(path)


def launch_streamlit_app(target: Path):
    ensure_safe_project(target)
    streamlit_page = (Path(__file__).parents[2] / "app" / "streamlit_entrypoint.py").expanduser().absolute()
    data_dir = target.expanduser().absolute().as_posix()
    sys.argv = ["streamlit", "run", streamlit_page.as_posix(), data_dir]

    # NOTE: we need to set PYTHONPATH for file watching
    python_path = Path(__file__).parents[2]
    environ["PYTHONPATH"] = (python_path).as_posix()

    server_args = [(python_path / "server" / "start_server.py").as_posix(), target.as_posix()]

    if app_config.is_dev:
        sys.argv.insert(3, "--server.fileWatcherType")
        sys.argv.insert(4, "auto")
        server_args.append("--reload")

    subprocess.Popen([sys.executable, *server_args])

    stcli.main()  # pylint: disable=no-value-for-parameter
