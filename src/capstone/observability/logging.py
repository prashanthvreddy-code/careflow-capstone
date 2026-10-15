"""Structured, redacted logs that carry the trace's correlation IDs. M7.

A log line is the "why" next to the trace's "where" (Day 4 slides: metrics, traces, logs share
one trace ID). Every CareFlow log record:
- is one JSON object (time, level, logger, message, trace_id, span_id, plus any extra fields),
- carries the trace_id / span_id of the span that was open when it was written, so a log line
  can be matched to its step in Langfuse or in runtime/traces/traces.jsonl,
- is member-number masked and cut to 300 characters, like the spans (no raw patient text).
"""
import json
import logging
from datetime import UTC, datetime

from capstone.observability.tracing import current_span
from capstone.schemas.state import mask_numbers

MAX_MESSAGE = 300
_STANDARD = set(vars(logging.makeLogRecord({}))) | {"message", "asctime"}


class CorrelationFilter(logging.Filter):
    """Adds the open span's IDs to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        span = current_span()
        record.trace_id = span.trace_id if span else None
        record.span_id = span.span_id if span else None
        return True


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = mask_numbers(record.getMessage())
        if len(message) > MAX_MESSAGE:
            message = message[: MAX_MESSAGE - 1] + "…"
        out = {"time": datetime.fromtimestamp(record.created, UTC).isoformat(timespec="milliseconds"),
               "level": record.levelname, "logger": record.name, "message": message,
               "trace_id": getattr(record, "trace_id", None), "span_id": getattr(record, "span_id", None)}
        extra = {k: v for k, v in vars(record).items() if k not in _STANDARD and k not in out}
        out.update(mask_numbers({k: v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
                                 for k, v in extra.items()}))
        if record.exc_info:
            out["error"] = record.exc_info[0].__name__ if record.exc_info[0] else None
        return json.dumps(out)


def get_logger(name: str = "careflow") -> logging.Logger:
    """A logger whose records get trace IDs; configure_logging() decides where they go."""
    logger = logging.getLogger(name)
    if not any(isinstance(f, CorrelationFilter) for f in logger.filters):
        logger.addFilter(CorrelationFilter())
    return logger


def configure_logging(level: int = logging.INFO, handler: logging.Handler | None = None) -> logging.Handler:
    """Send 'careflow' logs as JSON lines to `handler` (stderr by default). Called by entry points."""
    handler = handler or logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger("careflow")
    root.handlers = [h for h in root.handlers if not isinstance(h.formatter, JsonFormatter)] + [handler]
    root.setLevel(level)
    root.propagate = False
    return handler
