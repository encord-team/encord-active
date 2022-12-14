import sys
from pathlib import Path

from streamlit.web import cli as stcli


def launch_streamlit_app(target: Path):
    streamlit_page = (Path(__file__).parents[2] / "app" / "streamlit_entrypoint.py").expanduser().absolute()
    data_dir = target.expanduser().absolute().as_posix()
    sys.argv = ["streamlit", "run", streamlit_page.as_posix(), data_dir]

    sys.exit(stcli.main())  # pylint: disable=no-value-for-parameter
