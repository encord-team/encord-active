from encord_active.db.scripts.migrate_disk_to_db import migrate_disk_to_db
from encord_active.lib.project import ProjectFileStructure


def up(pfs: ProjectFileStructure) -> None:
    # FIXME: migration script execution should be ran.
    migrate_disk_to_db(pfs)
