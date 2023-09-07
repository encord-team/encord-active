import sys
from os import environ
from pathlib import Path


def start(path: Path, port: int, reload=False):
    from uvicorn import run

    environ["SERVER_START_PATH"] = path.as_posix()
    opts = {"reload": reload, "port": port, "host": environ.get("HOST")}
    if reload:
        server_watch_path = Path(__file__).parent.parent
        opts["reload_dirs"] = [server_watch_path.resolve().as_posix()]
    opts = {k: v for k, v in opts.items() if v is not None}
    environ.setdefault("API_URL", f"http://localhost:{port}")
    run("encord_active.server.main:app", **opts)


if __name__ == "__main__":
    start(Path(sys.argv[1]), port=8000, reload=("--reload" in sys.argv))
