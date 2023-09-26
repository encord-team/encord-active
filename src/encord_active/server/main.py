from encord_active.db.models import get_engine
from encord_active.server.app import get_app
from encord_active.server.settings import get_settings

settings = get_settings()
engine_path = settings.SERVER_START_PATH / "encord-active.sqlite"
engine = get_engine(engine_path, concurrent=True)

app = get_app(engine, settings)
