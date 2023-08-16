import uuid
from pathlib import Path

from sqlalchemy.engine import Engine
from sqlmodel import Session, select

from encord_active.analysis.config import create_analysis, default_torch_device
from encord_active.analysis.executor import SimpleExecutor
from encord_active.db.models import (
    Project,
    ProjectDataMetadata,
    ProjectDataUnitMetadata,
)


def migrate_db_to_new_metric_engine(
    engine: Engine,
    database_dir: Path,
    project_hash: uuid.UUID,
) -> None:
    print("=== Running Computation Engine ===")
    metric_engine = SimpleExecutor(create_analysis(default_torch_device()))
    with Session(engine) as sess:
        project = sess.exec(select(Project).where(Project.project_hash == project_hash)).first()
        if project is None:
            raise ValueError("Project does not exist in the database!!")
        data_metas = sess.exec(
            select(ProjectDataMetadata).where(ProjectDataMetadata.project_hash == project_hash)
        ).fetchall()
        data_units_metas = sess.exec(
            select(ProjectDataUnitMetadata).where(ProjectDataUnitMetadata.project_hash == project_hash)
        ).fetchall()

    # FIXME: catches exceptions for easy testing of subsets of partially broken
    try:
        res = metric_engine.execute_from_db(
            data_metas,
            data_units_metas,
            [],
            database_dir,
            project_hash,
            project.project_remote_ssh_key_path,
        )
        data_analytics, data_analytics_extra, annotation_analytics, annotation_analytics_extra, _ignore = res
        print("Success: NOTE => data is not stored anyway in the database yet!!! [FIXME]")
        # FIXME: upsert the values into the database.
        # FIXME: reduced embeddings, need to start working out train / serialize / load (joblib / pickle / other?)
        #  can be delayed for later compared to metric_engine

        # FIXME: delete & insert everything back into the database.
        # FIXME: re-calculate embedding reductions
        # FIXME: update embeddings
    except Exception as e:
        import traceback

        print(f"Caught exception: {e}")
        print(f"Stack trace: {traceback.format_exc()}")
        raise
    print("=== Finished Computation Engine ===")
