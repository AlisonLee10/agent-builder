import logging
import contextvars
import uuid
from datetime import datetime, timedelta
from pathlib import Path

LOGS_DIR = Path("logs")
RUNS_DIR = LOGS_DIR / "runs"

_run_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("run_id", default="--------")

_current_run_handler: logging.FileHandler | None = None

class _RunIDFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.run_id = _run_id_var.get()
        return True

def _configure() -> None:

    root = logging.getLogger()
    if root.handlers:
        return

    root.setLevel(logging.DEBUG)
    root.addFilter(_RunIDFilter())

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
    file_handler = logging.FileHandler(
        LOGS_DIR / f"{today}.log", encoding="utf-8"
    )

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
    _configure()
    return logging.getLogger(name)

def new_run_id() -> str:
    """
    Generate a new run ID, set it in the ContextVar,
    and open a per-run log file in logs/runs/.
    """
    global _current_run_handler
    _configure()

    run_id = str(uuid.uuid4())[:6]
    _run_id_var.set(run_id)

    # Open a dedicated log file for this run
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    _current_run_handler = logging.FileHandler(
        RUNS_DIR / f"{run_id}.log", encoding="utf-8"
    )
    _current_run_handler.setLevel(logging.DEBUG)
    _current_run_handler.setFormatter(logging.Formatter(
        fmt     = "%(asctime)s | %(levelname)-5s | %(name)-24s | %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
    ))
    _current_run_handler.addFilter(_RunIDFilter())
    logging.getLogger().addHandler(_current_run_handler)

    return run_id


def get_run_id() -> str:
    return _run_id_var.get()


def clear_run_id() -> None:
    """Close the per-run file handler and reset the ID."""
    global _current_run_handler
    if _current_run_handler:
        logging.getLogger().removeHandler(_current_run_handler)
        _current_run_handler.close()
        _current_run_handler = None
    _run_id_var.set("--------")

def set_level(level: str) -> None:
    target = getattr(logging, level.upper(), logging.INFO)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.StreamHandler) \
        and not isinstance(handler, logging.FileHandler):
            handler.setLevel(target)

# --- Log file utilities ---

def get_run_log_path(run_id: str) -> Path:
    """Return the path to a specific run's log file."""
    return RUNS_DIR / f"{run_id}.log"

def print_run_log(run_id: str) -> None:
    """Print the complete log for one run to the terminal."""
    path = get_run_log_path(run_id)
    if not path.exists():
        print(f"No log found for run {run_id}")
        return
    print(f"\n--- Log for run {run_id} ---")
    print(path.read_text(encoding="utf-8"))

def cleanup_old_logs(
    daily_keep_days: int = 7,
    run_keep_days: int = 30,
) -> None:
    """
    Delete old log files to prevent disk buildup.
    daily_keep_days: how many daily logs to keep (default 7)
    run_keep_days: how many days to keep per-run logs (default 30)
    """
    now = datetime.now()

    if LOGS_DIR.exists():
        for f in LOGS_DIR.glob("*log"):
            age = now - datetime.fromtimestamp(f.stat().st_ctime)
            if age > timedelta(days=daily_keep_days):
                f.unlink()

    if RUNS_DIR.exists():
        for f in RUNS_DIR.glob("*log"):
            age = now - datetime.fromtimestamp(f.stat().st_ctime)
            if age > timedelta(days=run_keep_days):
                f.unlink()