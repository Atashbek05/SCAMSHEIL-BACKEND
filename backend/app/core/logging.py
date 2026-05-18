"""
core/logging.py — Structured logging configuration using Loguru.

Call configure_logging() once at startup (done inside main.py).
Import `logger` from loguru anywhere else in the app.
"""

import sys
from loguru import logger


def configure_logging() -> None:
    logger.remove()  # remove default handler

    logger.add(
        sys.stdout,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | {message}",
        level="DEBUG",
    )

    logger.add(
        "logs/scamshield.log",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        level="INFO",
    )
