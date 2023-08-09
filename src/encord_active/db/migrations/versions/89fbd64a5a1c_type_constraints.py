"""type_constraints

Revision ID: 89fbd64a5a1c
Revises: ee71a9530e80
Create Date: 2023-08-08 11:22:44.356806+01:00

"""
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "89fbd64a5a1c"
down_revision = "ee71a9530e80"
branch_labels = None
depends_on = None

_annotation_type_migrate_mapping = [
    ("CLASSIFICATION", 0),
    ("BOUNDING_BOX", 1),
    ("ROT_BOUNDING_BOX", 2),
    ("POINT", 3),
    ("POLYLINE", 4),
    ("POLYGON", 5),
    ("SKELETON", 6),
    ("BITMASK", 7),
]


def upgrade() -> None:
    bind = op.get_bind()
    if bind.engine.name != 'sqlite':
        return  # SKIP - we process non-sqlite databases later.

    # ### commands auto generated by Alembic - please adjust! ###
    def upgrade_annotation_types(table: str) -> None:
        with op.batch_alter_table(table) as bop:
            bop.alter_column(
                "annotation_type",
                nullable=True,
                type_=sa.VARCHAR(),
                existing_type=sa.Enum(
                    "CLASSIFICATION",
                    "BOUNDING_BOX",
                    "ROT_BOUNDING_BOX",
                    "POINT",
                    "POLYLINE",
                    "POLYGON",
                    "SKELETON",
                    "BITMASK",
                    name="annotationtype",
                ),
            )
        for enum_str, enum_int in _annotation_type_migrate_mapping:
            op.execute(
                f"""
                UPDATE {table}
                SET annotation_type = '{enum_int}'
                WHERE annotation_type = '{enum_str}'
                """
            )
        bind = op.get_bind()
        if bind.engine.name == 'sqlite':
            with op.batch_alter_table(table) as bop:
                bop.alter_column(
                    "annotation_type",
                    nullable=False,
                    type_=sa.Integer(),
                    existing_type=sa.VARCHAR(),
                )
        else:
            op.execute(
                "ALTER TABLE active_project_analytics_annotation ALTER COLUMN annotation_type TYPE INTEGER "
                "USING annotation_type::integer"
            )

    upgrade_annotation_types("active_project_analytics_annotation")
    upgrade_annotation_types("active_project_prediction_analytics")
    upgrade_annotation_types("active_project_prediction_analytics_false_negatives")
    # ### end Alembic commands ###


def downgrade() -> None:
    bind = op.get_bind()
    if bind.engine.name != 'sqlite':
        return  # SKIP - we process non-sqlite databases later.

    # ### commands auto generated by Alembic - please adjust! ###
    def downgrade_annotation_types(table: str) -> None:
        with op.batch_alter_table(table) as bop:
            bop.alter_column(
                "annotation_type",
                nullable=False,
                type_=sa.VARCHAR(),
                existing_type=sa.Integer(),
            )
        for enum_str, enum_int in _annotation_type_migrate_mapping:
            op.execute(
                f"""
                UPDATE {table}
                SET annotation_type = '{enum_str}'
                WHERE annotation_type = '{enum_int}'
                """
            )
        with op.batch_alter_table(table) as bop:
            bop.alter_column(
                "annotation_type",
                nullable=True,
                type_=sa.VARCHAR(),
                existing_type=sa.Enum(
                    "CLASSIFICATION",
                    "BOUNDING_BOX",
                    "ROT_BOUNDING_BOX",
                    "POINT",
                    "POLYLINE",
                    "POLYGON",
                    "SKELETON",
                    "BITMASK",
                    name="annotationtype",
                ),
            )

    downgrade_annotation_types("active_project_analytics_annotation")
    downgrade_annotation_types("active_project_prediction_analytics")
    downgrade_annotation_types("active_project_prediction_analytics_false_negatives")
    # ### end Alembic commands ###
