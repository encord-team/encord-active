import json
from pathlib import Path
from typing import Optional

from fastapi.openapi.utils import get_openapi

from encord_active.server.app import get_app


def generate_openapi_fe_components() -> None:
    import os
    import subprocess
    from unittest.mock import MagicMock

    # Path lookup
    src_dir = Path(__file__).parent.parent.parent.resolve()
    fe_dir = src_dir.parent / "frontend_components" / "encord_active_components" / "frontend"
    openapi_json_path = fe_dir / "openapi.json"
    app = get_app(MagicMock(), MagicMock())
    # Output openapi
    with open(openapi_json_path, "w", encoding="UTF-8") as f:
        json.dump(
            get_openapi(
                title=app.title,
                version=app.version,
                openapi_version=app.openapi_version,
                description=app.description,
                routes=app.routes,
            ),
            f,
        )
    print("Generated openapi.json")
    # Locate openapi-generate-cli.sh
    openapi_gen_path = fe_dir / "node_modules" / ".bin" / "openapi-generator-cli"
    if not openapi_gen_path.exists():
        raise RuntimeError(
            f"openapi-generator-cli doesn't exist, make sure you ran `npm install` in the fronted directory"
        )
    openapi_fe_output = fe_dir / "src" / "openapi"
    prettier_module = fe_dir / "node_modules" / ".bin" / "prettier"
    command = [
        openapi_gen_path.as_posix(),
        "generate",
        "-i",
        openapi_json_path.as_posix(),
        "-o",
        openapi_fe_output.as_posix(),
        "-g",
        "typescript-axios",
        "--additional-properties=nullSafeAdditionalProps=true,supportsES6=true,useUnionTypes=true",
        "--enable-post-process-file",
    ]
    command_env = {
        **os.environ,
        "TS_POST_PROCESS_FILE": f"{prettier_module.as_posix()} --write",
        "OPENAPI_GENERATOR_VERSION": "7.0.0",
    }
    print(f"Running openapi generator: {' '.join(command)}")
    subprocess.run(" ".join(command), shell=True, check=True, env=command_env, cwd=fe_dir)


if __name__ == "__main__":
    generate_openapi_fe_components()
