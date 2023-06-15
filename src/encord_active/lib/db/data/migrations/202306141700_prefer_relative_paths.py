from encord_active.lib.common.data_utils import file_path_to_url, url_to_file_path
from encord_active.lib.db.connection import PrismaConnection
from encord_active.lib.project import ProjectFileStructure


def up(pfs: ProjectFileStructure) -> None:
    with PrismaConnection(pfs) as conn:
        data_units = conn.dataunit.find_many(
            where={
                "data_uri": {
                    "startswith": "file://",
                }
            }
        )
        with conn.batch_() as batcher:
            for data_unit in data_units:
                if data_unit.data_uri is not None:
                    path = url_to_file_path(
                        data_unit.data_uri,
                        project_dir=pfs.project_dir,
                    )
                    if path is None:
                        continue
                    new_uri = file_path_to_url(
                        path,
                        project_dir=pfs.project_dir,
                    )
                    if data_unit.data_uri != new_uri:
                        batcher.dataunit.update(
                            data={
                                "data_uri": new_uri,
                            },
                            where={
                                "data_hash_frame": {
                                    "data_hash": data_unit.data_hash,
                                    "frame": data_unit.frame,
                                },
                            },
                        )
            batcher.commit()
