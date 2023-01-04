import sys
from os import environ
from pathlib import Path

from streamlit.web import cli as stcli


def launch_streamlit_app(target: Path):
    streamlit_page = (Path(__file__).parents[2] / "app" / "streamlit_entrypoint.py").expanduser().absolute()
    data_dir = target.expanduser().absolute().as_posix()
    sys.argv = ["streamlit", "run", streamlit_page.as_posix(), data_dir]

    # NOTE: we need to set PYTHONPATH for file watching
    environ["PYTHONPATH"] = (Path(__file__).parents[2]).as_posix()
    sys.exit(stcli.main())  # pylint: disable=no-value-for-parameter
