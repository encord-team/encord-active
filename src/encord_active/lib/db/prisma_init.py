import filecmp
from pathlib import Path

from prisma import __file__ as prisma_module_file
from prisma.cli.prisma import run

from encord_active.lib.common.decorators import silence_stdout

prisma_run = silence_stdout(run)

GENERATED_PRISMA_SCHEMA_FILE = Path(prisma_module_file).parent / "schema.prisma"
PRISMA_SCHEMA_FILE = Path(__file__).parent / "schema.prisma"


def ensure_prisma_db(db_path: Path):
    env = {"MY_DATABASE_URL": f"file:{db_path}"}
    failure = prisma_run(["migrate", "deploy", f"--schema={PRISMA_SCHEMA_FILE}"], env=env)
    # TODO: remove me before release
    if failure:
        prisma_run(["migrate", "reset", "--force", "--skip-generate", f"--schema={PRISMA_SCHEMA_FILE}"], env=env)


def generate_prisma_client():
    prisma_run(["generate", f"--schema={PRISMA_SCHEMA_FILE}"])


def did_schema_change():
    return not filecmp.cmp(PRISMA_SCHEMA_FILE, GENERATED_PRISMA_SCHEMA_FILE)
