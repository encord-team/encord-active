import filecmp
from logging import getLogger
from pathlib import Path
from typing import Set

from prisma import __file__ as prisma_module_file
from prisma.cli.prisma import run

from encord_active.lib.common.decorators import silence_stdout

prisma_run = silence_stdout(run)

logger = getLogger("Prisma")

GENERATED_PRISMA_SCHEMA_FILE = Path(prisma_module_file).parent / "schema.prisma"
PRISMA_SCHEMA_FILE = Path(__file__).parent / "schema.prisma"

CACHE_ENSURE_PRISMA_DB: Set[str] = set()


def ensure_prisma_db(db_path: Path):
    db_key = db_path.absolute().as_posix()
    if db_key in CACHE_ENSURE_PRISMA_DB:
        return

    env = {"MY_DATABASE_URL": f"file:{db_path}"}
    failure = prisma_run(["migrate", "deploy", f"--schema={PRISMA_SCHEMA_FILE}"], env=env)
    # TODO: remove me before release
    if failure:
        prisma_run(["migrate", "reset", "--force", "--skip-generate", f"--schema={PRISMA_SCHEMA_FILE}"], env=env)

    CACHE_ENSURE_PRISMA_DB.add(db_key)


def generate_prisma_client():
    logger.info("Regenerating prisma DB, please re-run the command if an issue is detected")
    prisma_run(["generate", f"--schema={PRISMA_SCHEMA_FILE}"])


def did_schema_change():
    return not filecmp.cmp(PRISMA_SCHEMA_FILE, GENERATED_PRISMA_SCHEMA_FILE)
