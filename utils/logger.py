"""
utils/logger.py
---------------
Structured logger for the pipeline.

Provides a single `get_logger(name)` factory so every module gets a
consistently formatted logger without repeating setup boilerplate.
Swap the handler or formatter here once to affect the whole system.
"""

import logging
import sys


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Return a named logger with a stdout StreamHandler.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    level:
        Default log level (INFO). Override per-module if needed.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(level)
    return logger
