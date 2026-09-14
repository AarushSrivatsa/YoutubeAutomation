"""
logging_config.py — one place to configure logging for the whole app.

Call configure_logging() once, as early as possible, in every process
entrypoint (currently just main.py, since background jobs run in-process).
Every other module just does:

    import logging
    logger = logging.getLogger(__name__)

and logs normally — the format/level below applies everywhere so every
step of every job shows up in the terminal with a timestamp and origin.
"""
from __future__ import annotations
import logging
import os
import sys

_CONFIGURED = False


def configure_logging(level: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_level = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    numeric_level = getattr(logging, log_level, logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)-32s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.handlers.clear()
    root.addHandler(handler)

    # Quiet down noisy third-party loggers unless we're actually debugging
    if numeric_level > logging.DEBUG:
        for noisy in ("httpx", "httpcore", "urllib3", "arq", "asyncio"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True
    logging.getLogger(__name__).info("Logging configured at level %s", log_level)
