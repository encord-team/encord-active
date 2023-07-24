from encord_active.db.scripts.migrate_disk_to_db import migrate_disk_to_db
from encord_active.lib.common.data_utils import file_path_to_url, url_to_file_path
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.project import ProjectFileStructure


def up(pfs: ProjectFileStructure) -> None:
    # Fix issues caused by absolute paths being used for paths to the project.
    # Upgrade file paths to relative paths wherever it is possible.
    with PrismaConnection(pfs) as conn:
        data_units = conn.dataunit.find_many()
        with conn.batch_() as batch:
            for data_unit in data_units:
                if data_unit.data_uri is None:
                    continue
                data_uri_path = url_to_file_path(data_unit.data_uri, pfs.project_dir)
                if data_uri_path is None:
                    continue
                data_uri = file_path_to_url(data_uri_path, pfs.project_dir)
                if data_uri != data_unit.data_uri:
                    batch.dataunit.update(
                        data={
                            "data_uri": data_uri,
                        },
                        where={
                            "data_hash_frame": {
                                "data_hash": data_unit.data_hash,
                                "frame": data_unit.frame,
                            }
                        },
                    )
            batch.commit()
    # Run database migration script
    migrate_disk_to_db(pfs)
