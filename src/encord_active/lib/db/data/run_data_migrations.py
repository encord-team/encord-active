import importlib.util
import logging
import sys
from pathlib import Path

from encord_active.lib.project.metadata import fetch_project_meta, update_project_meta
from encord_active.lib.project.project_file_structure import ProjectFileStructure

logger = logging.getLogger("Data Migrations")


def get_timestamp_of_migration_file(file: Path):
    return int(file.name.split("_")[0])


def run_data_migrations(pfs: ProjectFileStructure):
    project_meta = fetch_project_meta(pfs.project_dir)
    last_migration_timestemp = project_meta.get("data_version") or 0

    all_migrations = list((Path(__file__).parent / "migrations").glob("*.py"))
    migrations_to_run = [
        migration
        for migration in all_migrations
        if get_timestamp_of_migration_file(migration) > last_migration_timestemp
    ]
    migrations_to_run.sort(key=lambda file: get_timestamp_of_migration_file(file))

    if not migrations_to_run:
        return

    for migration_file in migrations_to_run:
        module_name = migration_file.name
        logger.info(f"Running migration: {module_name}")
        # migration = importlib.import_module(migration_file.absolute().as_posix())

        spec = importlib.util.spec_from_file_location(module_name, migration_file)
        if spec and spec.loader:
            migration = importlib.util.module_from_spec(spec)
            sys.modules["module.name"] = migration
            spec.loader.exec_module(migration)
            migration.up(pfs)

    # FIXME: re-enable this before merging (just makes debugging much easier)
    # project_meta["data_version"] = get_timestamp_of_migration_file(migrations_to_run[-1])
    # update_project_meta(pfs.project_dir, project_meta)
