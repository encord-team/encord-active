import sys
from os import environ
from pathlib import Path


def start(path: Path, reload=False):
    from uvicorn import run

    environ["SERVER_START_PATH"] = path.as_posix()
    opts = {"reload": reload, "port": environ.get("PORT"), "host": environ.get("HOST")}
    if reload:
        server_watch_path = Path(__file__).parent.parent
        opts["reload_dirs"] = [server_watch_path.resolve().as_posix()]
    opts = {k: v for k, v in opts.items() if v is not None}
    run("encord_active.server.main:app", **opts)


if __name__ == "__main__":
    start(Path(sys.argv[1]), reload=("--reload" in sys.argv))
