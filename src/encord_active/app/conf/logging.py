import logging


def setup_logging():
    disabled_loggers = ["faiss.loader"]

    for logger_name in disabled_loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.ERROR)
