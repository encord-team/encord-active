import sys

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")
