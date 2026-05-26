import logging
import os
from datetime import datetime
from pathlib import Path

LOGS_DIR = Path("logs")


def _configure() -> None:
    """Set up root logger once. Subsequent calls are no-ops."""

    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)

    # -- Terminal handler - INFO and above, concise -------
    terminal = logging.StreamHandler()
    terminal.setLevel(logging.INFO)
    terminal.setFormatter(logging.Formatter(
        fmt = "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt = "%H:%M:%S"
    ))

    # -- File handler - DEBUG and above, detailed -------
    LOGS_DIR.mkdir(exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOGS_DIR / f"{today}.log"

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt = "%(asctime)s | %(levelname)-5s | %(name)-24s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    ))

    root.addHandler(terminal)
    root.addHandler(file_handler)

    # -- Silence noisy third-party loggers ------
    for noisy in ("httpx", "httpcore", "openai", "langchain", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    Call once per module at the top of the file:
        log = get_logger(__name__)    
    """
    _configure()
    return logging.getLogger(name)


def set_level(level: str) -> None:
    """
    Change terminal verbosity at runtime.
    set_level("DEBUG") turns on full detail.
    set_level("INFO") returns to normal.
    """
    target = getattr(logging, level.upper(), logging.INFO)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) \
        and not isinstance(handler, logging.FileHandler):
            handler.setLevel(target)