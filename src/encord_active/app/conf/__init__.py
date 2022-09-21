from os import environ

from dotenv import load_dotenv

from encord_active.app.conf.logging import setup_logging

load_dotenv()
setup_logging()

FRONTEND = environ.get("FRONTEND", "http://localhost:5173")
