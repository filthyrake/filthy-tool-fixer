"""Structured logging setup with request ID context."""

from __future__ import annotations

import contextvars
import logging
import uuid

import structlog

from filthyllm.config import settings

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "request_id", default=""
)


def add_request_id(
    logger: structlog.types.WrappedLogger,
    method_name: str,
    event_dict: structlog.types.EventDict,
) -> structlog.types.EventDict:
    rid = request_id_var.get()
    if rid:
        event_dict["request_id"] = rid
    return event_dict


def generate_request_id() -> str:
    return uuid.uuid4().hex[:12]


def setup_logging() -> None:
    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        add_request_id,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)
