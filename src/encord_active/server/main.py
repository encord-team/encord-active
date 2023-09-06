from fastapi.security import OAuth2PasswordBearer

from encord_active.db.models import get_engine
from encord_active.server.app import get_app
from encord_active.server.settings import get_settings


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token", auto_error=False)

engine_path = get_settings().SERVER_START_PATH / "encord-active.sqlite"
engine = get_engine(engine_path, concurrent=True)

app = get_app(engine, oauth2_scheme)
