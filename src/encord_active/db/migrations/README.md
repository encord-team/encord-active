# Generic single-database configuration.

#### Creating a new migration:

First ensure that alembic.ini has sqlalchemy.url pointed to a
database that can be used for verifying migrations.

Ensure that the database is updated to head (should happen automatically
if this database has been used for encord-active development).

> alembic upgrade head

Then generate new version file, this will attempt to generate a new migration
automatically from the schema defined in src/db/models.py - it is normally correct
but for the migration to work correctly on all database types we want to suppor.

> alembic revision —autogenerate -m "{migration_identifier}”
