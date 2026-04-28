from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer


class DebouncedEventHandler(FileSystemEventHandler):
    def __init__(self, callback: Callable[[], None], debounce_seconds: float = 1.0):
        super().__init__()
        self._callback = callback
        self._debounce_seconds = debounce_seconds
        self._timer: Optional[threading.Timer] = None
        self._lock = threading.Lock()

    def _schedule(self) -> None:
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            self._timer = threading.Timer(self._debounce_seconds, self._callback)
            self._timer.daemon = True
            self._timer.start()

    def on_modified(self, event):
        if not event.is_directory:
            self._schedule()

    def on_created(self, event):
        if not event.is_directory:
            self._schedule()

    def on_moved(self, event):
        if not event.is_directory:
            self._schedule()

    def on_deleted(self, event):
        if not event.is_directory:
            self._schedule()


def watch(vault: Path, callback: Callable[[], None]) -> None:
    observer = Observer()
    handler = DebouncedEventHandler(callback)
    observer.schedule(handler, str(vault), recursive=True)
    observer.start()
    try:
        while True:
            time.sleep(0.5)
    finally:
        observer.stop()
        observer.join()
