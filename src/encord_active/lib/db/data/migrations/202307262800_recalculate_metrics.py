from encord_active.lib.project import ProjectFileStructure


def up(_pfs: ProjectFileStructure) -> None:
    # FIXME: this migration should be considered 'in-progress'
    # database_dir = pfs.project_dir.parent.expanduser().resolve()
    # project_hash = uuid.UUID(pfs.load_project_meta()["project_hash"])
    # path = database_dir / "encord-active.sqlite"
    # engine = get_engine(path)
    # migrate_db_to_new_metric_engine(engine, database_dir, project_hash)
    pass  # NO - OP
