import sys
from os import environ
from pathlib import Path


def start(path: Path, reload=False):
    from uvicorn import run

    environ["SERVER_START_PATH"] = path.as_posix()
    run("encord_active.server.main:app", reload=reload, host="0.0.0.0", port=8502)


if __name__ == "__main__":
    start(Path(sys.argv[1]), reload=("--reload" in sys.argv))
