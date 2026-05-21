import sys
import threading
from contextlib import contextmanager


@contextmanager
def show_progress(message: str = "Working", interval: float = 0.4):
    """Show animated dots on one terminal line while a long task runs."""
    stop = threading.Event()

    def _animate() -> None:
        dots = 0
        while not stop.wait(interval):
            dots = (dots + 1) % 4
            trail = "." * dots + " " * (3 - dots)
            sys.stdout.write(f"\r{message}{trail}")
            sys.stdout.flush()

    thread = threading.Thread(target=_animate, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=interval + 1)
        sys.stdout.write("\r" + " " * (len(message) + 4) + "\r")
        sys.stdout.flush()
