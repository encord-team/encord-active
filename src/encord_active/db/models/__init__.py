import os
import typing
from pathlib import Path
from sqlite3 import Connection as SqliteConnection
from typing import Set, Union

if typing.TYPE_CHECKING:
    from psycopg2.extensions import connection as pg_connection
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlmodel import create_engine

from .project import *
from .project_annotation_analytics import *
from .project_annotation_analytics_derived import *
from .project_annotation_analytics_extra import *
from .project_annotation_analytics_reduced import *
from .project_collaborator import *
from .project_data_analytics import *
from .project_data_analytics_derived import *
from .project_data_analytics_extra import *
from .project_data_analytics_reduced import *
from .project_data_metadata import *
from .project_data_unit_metadata import *
from .project_embedding_index import *
from .project_embedding_reduction import *
from .project_import_metadata import *
from .project_prediction import *
from .project_prediction_analytics import *
from .project_prediction_analytics_derived import *
from .project_prediction_analytics_extra import *
from .project_prediction_analytics_false_negatives import *
from .project_prediction_analytics_reduced import *
from .project_prediction_data_metadata import *
from .project_prediction_data_unit_metadata import *
from .project_tag import *
from .project_tagged_annotation import *
from .project_tagged_data import *
from .project_tagged_prediction import *

# FIXME: metrics should be inline predictions or separate (need same keys for easy comparison)??


_init_metadata: Set[str] = set()


@event.listens_for(Engine, "connect")
def set_sqlite_fk_pragma(
    dbapi_connection: Union["pg_connection", SqliteConnection],
    connection_record: None,
) -> None:
    # For sqlite - enable foreign_keys support
    #  (otherwise we need to use complex delete statements)
    if isinstance(dbapi_connection, SqliteConnection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def get_engine(
    path: Path,
    concurrent: bool = False,
    use_alembic: bool = True,
) -> Engine:
    override_db = os.environ.get("ENCORD_ACTIVE_DATABASE", None)
    database_echo = os.environ.get("ENCORD_ACTIVE_DATABASE_ECHO", "0") == "1"
    create_db_schema = os.environ.get("ENCORD_ACTIVE_DATABASE_SCHEMA_UPDATE", "1")

    path = path.expanduser().resolve()
    engine_url = override_db if override_db is not None else f"sqlite:///{path}"
    connect_args = {"check_same_thread": False} if concurrent and engine_url.startswith("sqlite:/") else {}

    # Create the engine connection
    print(f"Connection to database: {engine_url}")
    engine = create_engine(engine_url, connect_args=connect_args, echo=database_echo)
    path_key = path.as_posix()
    if path_key not in _init_metadata and create_db_schema == "1":
        if use_alembic:
            # Execute alembic config
            import encord_active.db as alembic_file

            alembic_cwd = Path(alembic_file.__file__).expanduser().resolve().parent
            alembic_args = [
                "alembic",
                "--raiseerr",
                "-x",
                f"dbPath={engine_url}",
                "upgrade",
                "head",
            ]
            # How to run alembic via subprocess if running locally with temp cwd change
            # ends up causing issues:
            # res_code = -1
            # if os.environ.get('ENCORD_ACTIVE_USE_SUBPROC_ALEMBIC', '0') == '1':
            #     # optional mode that doesn't change the cwd.
            #     # not used by default.
            #     import subprocess
            #     res_code = subprocess.Popen(alembic_args, cwd=alembic_cwd).wait()
            # if res_code != 0:
            # Run via CWD switch.
            current_cwd = os.getcwd()
            try:
                os.chdir(alembic_cwd)
                import alembic.config

                alembic.config.main(argv=alembic_args[1:])
            finally:
                os.chdir(current_cwd)
        else:
            SQLModel.metadata.create_all(engine)
        _init_metadata.add(path_key)
    return engine
