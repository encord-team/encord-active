import pandas as pd
from encord.objects import OntologyStructure
from sqlalchemy import MetaData, Table, create_engine, select, text
from sqlalchemy.engine import Engine


def get_active_engine(path_to_db: str) -> Engine:
    return create_engine(f"sqlite:///{path_to_db}")


class ActiveProject:
    def __init__(self, engine: Engine, project_name: str):
        self._engine = engine
        self._metadata = MetaData(bind=self._engine)
        self._project_name = project_name

        active_project = Table("active_project", self._metadata, autoload_with=self._engine)
        stmt = select(active_project.c.project_hash).where(active_project.c.project_name == f"{self._project_name}")

        with self._engine.connect() as connection:
            result = connection.execute(stmt).fetchone()

            if result is not None:
                self._project_hash = result[0]
            else:
                self._project_hash = None

    def get_ontology(self) -> OntologyStructure:
        active_project = Table("active_project", self._metadata, autoload_with=self._engine)

        stmt = select(active_project.c.project_ontology).where(active_project.c.project_hash == f"{self._project_hash}")

        with self._engine.connect() as connection:
            result = connection.execute(stmt).fetchone()

            if result is not None:
                ontology = result[0]
            else:
                ontology = None

        return OntologyStructure.from_dict(ontology)

    def get_prediction_metrics(self) -> pd.DataFrame:
        active_project_prediction = Table("active_project_prediction", self._metadata, autoload_with=self._engine)
        stmt = select(active_project_prediction.c.prediction_hash).where(
            active_project_prediction.c.project_hash == f"{self._project_hash}"
        )

        with self._engine.connect() as connection:
            result = connection.execute(stmt).fetchone()

            if result is not None:
                prediction_hash = result[0]
            else:
                prediction_hash = None

        active_project_prediction_analytics = Table(
            "active_project_prediction_analytics", self._metadata, autoload_with=self._engine
        )

        stmt = select(
            [
                active_project_prediction_analytics.c.feature_hash,
                active_project_prediction_analytics.c.metric_area,
                active_project_prediction_analytics.c.metric_area_relative,
                active_project_prediction_analytics.c.metric_aspect_ratio,
                active_project_prediction_analytics.c.metric_brightness,
                active_project_prediction_analytics.c.metric_contrast,
                active_project_prediction_analytics.c.metric_sharpness,
                active_project_prediction_analytics.c.metric_red,
                active_project_prediction_analytics.c.metric_green,
                active_project_prediction_analytics.c.metric_blue,
                active_project_prediction_analytics.c.metric_label_border_closeness,
                active_project_prediction_analytics.c.metric_label_confidence,
                text(
                    """CASE
                        WHEN feature_hash == match_feature_hash THEN 1
                        ELSE 0
                    END AS true_positive
                """
                ),
            ]
        ).where(active_project_prediction_analytics.c.prediction_hash == prediction_hash)

        with self._engine.begin() as conn:
            df = pd.read_sql(stmt, conn)

        return df
