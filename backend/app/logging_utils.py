"""Logging configuration and middleware helpers for the FastAPI app."""
from __future__ import annotations

import logging
import sys
import time
from typing import Optional

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

RESET = "\x1b[0m"
COLOR_MAP = {
    logging.DEBUG: "\x1b[36m",  # cyan
    logging.INFO: "\x1b[34m",   # blue
    logging.WARNING: "\x1b[33m",  # yellow
    logging.ERROR: "\x1b[31m",   # red
    logging.CRITICAL: "\x1b[31m",  # red
}


class ColorFormatter(logging.Formatter):
    """Formatter that adds ANSI colors based on the log level."""

    def format(self, record: logging.LogRecord) -> str:  # noqa: D401 - inherited docstring sufficient
        base = super().format(record)
        color = COLOR_MAP.get(record.levelno)
        if not color:
            return base
        return f"{color}{base}{RESET}"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging with color support.

    This is safe to call multiple times—the configuration will only be applied once.
    """

    root = logging.getLogger()
    if getattr(root, "_archeosensei_logging", False):  # type: ignore[attr-defined]
        return

    formatter = ColorFormatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    if root.handlers:
        for handler in root.handlers:
            handler.setFormatter(formatter)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        root.addHandler(handler)

    root.setLevel(level)
    setattr(root, "_archeosensei_logging", True)


class APILoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs every API request and highlights error responses."""

    def __init__(self, app, logger: Optional[logging.Logger] = None) -> None:
        super().__init__(app)
        self.logger = logger or logging.getLogger("archeosensei.api")

    async def dispatch(self, request: Request, call_next) -> Response:  # type: ignore[override]
        method = request.method
        path = request.url.path
        self.logger.info("➡️  %s %s", method, path)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self.logger.exception("❌ %s %s raised an unhandled exception after %.1f ms", method, path, elapsed_ms)
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        log_message = "⬅️  %s %s -> %s (%.1f ms)"

        if response.status_code >= 400:
            self.logger.error(log_message, method, path, response.status_code, elapsed_ms)
        else:
            self.logger.info(log_message, method, path, response.status_code, elapsed_ms)
        return response
