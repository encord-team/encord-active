from encord_active.db.models import get_engine
from encord_active.server.settings import get_settings

engine_path = get_settings().SERVER_START_PATH / "encord-active.sqlite"
engine = get_engine(engine_path, concurrent=True)
