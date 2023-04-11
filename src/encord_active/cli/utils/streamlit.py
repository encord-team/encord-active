import http.server
import os
import socketserver
import sys
import threading
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

PORT = 8000


class QuietServer(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        pass


def run_file_server(target: Path):
    os.chdir(target.resolve().as_posix())
    with socketserver.TCPServer(("", PORT), QuietServer) as httpd:
        print("Server started at localhost:" + str(PORT))
        httpd.serve_forever()


def launch_streamlit_app(target: Path):
    ensure_safe_project(target)
    streamlit_page = (Path(__file__).parents[2] / "app" / "streamlit_entrypoint.py").expanduser().absolute()
    data_dir = target.expanduser().absolute().as_posix()
    sys.argv = ["streamlit", "run", streamlit_page.as_posix(), data_dir]

    # NOTE: we need to set PYTHONPATH for file watching
    environ["PYTHONPATH"] = (Path(__file__).parents[2]).as_posix()

    if app_config.is_dev:
        sys.argv.insert(3, "--server.fileWatcherType")
        sys.argv.insert(4, "auto")

    # server_thread = threading.Thread(target=run_file_server, args=(target,))
    # server_thread.start()

    stcli.main()  # pylint: disable=no-value-for-parameter
