from typing import Optional
from uuid import UUID
from sqlmodel import Field, SQLModel, create_engine


class Project(SQLModel, table=True):
    project_hash: UUID = Field(default=None, primary_key=True)
    project_name: str

class ProjectData(SQLModel, table=True):



class ProjectMetric(SQLModel, table=True):
    metric_id: int = Field(default=)