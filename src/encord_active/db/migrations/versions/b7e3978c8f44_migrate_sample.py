"""“migrate_sample”

Revision ID: b7e3978c8f44
Revises: bcfdfc2f498a
Create Date: 2023-08-09 09:59:46.491945+01:00

"""
import sqlalchemy as sa
import sqlmodel
from alembic import op


# revision identifiers, used by Alembic.
revision = "b7e3978c8f44"
down_revision = "bcfdfc2f498a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column("active_project_analytics_data", sa.Column("metric_video_red_density", sa.Float(), nullable=True))
    op.create_index(
        "active_data_project_hash_metric_video_red_density_index",
        "active_project_analytics_data",
        ["project_hash", "metric_video_red_density"],
        unique=False,
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_index("active_data_project_hash_metric_video_red_density_index", table_name="active_project_analytics_data")
    op.drop_column("active_project_analytics_data", "metric_video_red_density")
    # ### end Alembic commands ###
