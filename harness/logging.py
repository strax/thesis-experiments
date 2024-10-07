import logging
import sys
from typing import cast, Iterable

import structlog
from structlog.stdlib import BoundLogger

def format_log_level(_: object, value: object):
    level = cast(str, value)
    match level:
        case "debug": return "[*]"
        case "info": return "[+]"
        case "warning": return "[-]"
        case "error" | "critical": return "[!]"

def format_task(_: object, value: object):
    if not isinstance(value, str) and isinstance(value, Iterable):
        value = "/".join(map(str, value))
    return f"{value}:"

renderer = structlog.dev.ConsoleRenderer(
    columns=[
        structlog.dev.Column(
            "level",
            format_log_level
        ),
        structlog.dev.Column(
            "task",
            format_task
        ),
        structlog.dev.Column(
            "event",
            lambda _, value: value
        ),
        structlog.dev.Column(
            "",
            structlog.dev.KeyValueColumnFormatter(
                key_style="",
                value_style="",
                reset_style="",
                value_repr=str
            )
        ),
    ]
)

def configure_logging(verbose = False):
    structlog.configure_once(
        processors=[
            structlog.processors.add_log_level,
            renderer
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG if verbose else logging.INFO),
        logger_factory=structlog.PrintLoggerFactory()
    )

type Logger = BoundLogger

def get_logger() -> Logger:
    return structlog.stdlib.get_logger()

__all__ = ["configure_logging", "get_logger", "Logger"]
