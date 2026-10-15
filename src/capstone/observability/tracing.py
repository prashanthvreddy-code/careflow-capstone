"""Tracing: one trace per request, one span per step, tool call and LLM call. M7.

Decided in M7 phase 2:
- Two sinks: Langfuse Cloud (when the runner switches it on) AND a local JSONL file
  (runtime/traces/traces.jsonl) with the same spans. Tests read only the local file.
- METADATA ONLY. A span carries names, timings, status, the model that actually answered,
  token counts, attempts, fallback/breaker events, route/outcome and record_id/patient_ref.
  Never prompt or completion text: keys that usually hold text (prompt, messages, content,
  raw_text, answer, ...) are dropped, every string is member-number masked and cut to 120
  characters. Langfuse also gets the same mask as a second line of defence ("redact before it
  leaves the process", Day 4 slides).
- Tracing must never break a request: a sink that fails is switched off for the rest of the
  run and the request carries on. Langfuse fails silently on bad keys, so `check()` asks it.
- Nesting follows the call stack (a contextvar), like Lab A: no span objects are passed around.
- Spans are written when they END, so a child line appears before its parent's line.
"""
import contextvars
import json
import threading
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from capstone.schemas.state import mask_numbers

Kind = Literal["span", "generation", "tool", "agent", "chain"]
TEXT_KEYS = {"prompt", "prompts", "messages", "message", "content", "text", "raw_text", "completion",
             "answer", "input", "output", "note", "notes", "reason_text", "body"}
MAX_STR = 120
MAX_LIST = 20

_current: contextvars.ContextVar["Span | None"] = contextvars.ContextVar("careflow_span", default=None)


def clean_attrs(attrs: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Keep only short, masked metadata. Returns (kept, dropped keys)."""
    kept, dropped = {}, []
    for key, value in attrs.items():
        if key.lower() in TEXT_KEYS:
            dropped.append(key)
            continue
        v = _clean_value(value)
        if v is _DROP:
            dropped.append(key)
        else:
            kept[key] = v
    return kept, dropped


_DROP = object()


def _clean_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        s = mask_numbers(value)
        return s if len(s) <= MAX_STR else s[: MAX_STR - 1] + "…"
    if isinstance(value, (list, tuple)):
        items = [_clean_value(v) for v in list(value)[:MAX_LIST]]
        return _DROP if any(i is _DROP for i in items) else items
    if isinstance(value, dict):                         # only small dicts of numbers, e.g. token usage
        ok = all(v is None or isinstance(v, (bool, int, float)) for v in value.values())
        return {str(k): v for k, v in value.items()} if ok else _DROP
    return _DROP


class Span:
    """One unit of work. Use tracer.span(...) to make one."""

    def __init__(self, tracer: "Tracer", name: str, kind: Kind, parent: "Span | None", attrs: dict):
        self.tracer, self.name, self.kind, self.parent = tracer, name, kind, parent
        self.trace_id = parent.trace_id if parent else uuid.uuid4().hex
        self.span_id = uuid.uuid4().hex[:16]
        self.attrs: dict[str, Any] = {}
        self.dropped: list[str] = []
        self.events: list[dict] = []
        self.status, self.error = "ok", None
        self.start = time.time()
        self._t0 = time.perf_counter()
        self.duration_ms: float | None = None
        self._lf = None                                   # the Langfuse observation, if any
        self.set(**attrs)

    def set(self, **attrs) -> None:
        kept, dropped = clean_attrs(attrs)
        self.attrs.update(kept)
        self.dropped += [d for d in dropped if d not in self.dropped]

    def event(self, event_name: str, /, **attrs) -> None:
        """Something that happened inside the span: a retry, a breaker change, a fallback."""
        kept, _ = clean_attrs(attrs)
        self.events.append({**kept, "name": event_name, "at_ms": round((time.perf_counter() - self._t0) * 1000, 1)})

    def fail(self, error: BaseException) -> None:
        self.status = "error"
        self.error = {"type": type(error).__name__, "message": _clean_value(str(error))}

    def record(self) -> dict:
        return {"trace_id": self.trace_id, "span_id": self.span_id,
                "parent_id": self.parent.span_id if self.parent else None,
                "name": self.name, "kind": self.kind,
                "start": datetime.fromtimestamp(self.start, UTC).isoformat(timespec="milliseconds"),
                "duration_ms": self.duration_ms, "status": self.status, "error": self.error,
                "attrs": self.attrs, "events": self.events, "dropped_keys": self.dropped}


class LocalSink:
    """Appends one JSON line per finished span."""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()

    def write(self, record: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock, self.path.open("a") as f:
            f.write(json.dumps(record) + "\n")


class LangfuseSink:
    """Sends the same metadata spans to Langfuse. Built only when the runner asks for it."""

    def __init__(self, client=None):
        if client is None:
            from langfuse import Langfuse
            client = Langfuse(mask=lambda *, data, **_: mask_numbers(data),
                              should_export_span=lambda s: (s.instrumentation_scope is not None and
                                                            s.instrumentation_scope.name.startswith("langfuse")))
        self.client = client

    def check(self) -> bool:
        try:
            return bool(self.client.auth_check())
        except Exception:  # noqa: BLE001 - a bad key or no network = not available
            return False

    def open(self, span: Span):
        as_type = "generation" if span.kind == "generation" else span.kind
        cm = self.client.start_as_current_observation(name=span.name, as_type=as_type, metadata=span.attrs)
        obs = cm.__enter__()
        tags = None
        if span.parent is None:                    # trace-level tags, inherited by every child span
            from langfuse import propagate_attributes
            tags = propagate_attributes(session_id=span.attrs.get("session_id"),
                                        user_id=span.attrs.get("patient_ref"),
                                        tags=["careflow", "m7"], trace_name=span.name)
            tags.__enter__()
        return cm, obs, tags

    def close(self, span: Span, handle) -> None:
        cm, obs, tags = handle
        extra = {}
        if span.kind == "generation":
            extra = {"model": span.attrs.get("model"), "usage_details": span.attrs.get("usage") or None}
        obs.update(metadata={**span.attrs, "events": span.events, "status": span.status},
                   level="ERROR" if span.status == "error" else "DEFAULT",
                   status_message=(span.error or {}).get("type"), **{k: v for k, v in extra.items() if v})
        if tags is not None:
            tags.__exit__(None, None, None)
        cm.__exit__(None, None, None)

    def flush(self) -> None:
        self.client.flush()


class Tracer:
    def __init__(self, local: LocalSink | None = None, cloud: LangfuseSink | None = None):
        self.local, self.cloud = local, cloud
        self.sink_errors: list[str] = []

    @contextmanager
    def span(self, name: str, kind: Kind = "span", **attrs) -> Iterator[Span]:
        span = Span(self, name, kind, _current.get(), attrs)
        token = _current.set(span)
        handle = self._safe("open", lambda: self.cloud.open(span)) if self.cloud else None
        try:
            yield span
        except BaseException as e:
            span.fail(e)
            raise
        finally:
            span.duration_ms = round((time.perf_counter() - span._t0) * 1000, 1)
            _current.reset(token)
            if self.cloud and handle is not None:
                self._safe("close", lambda: self.cloud.close(span, handle))
            if self.local:
                self._safe("local", lambda: self.local.write(span.record()))

    def _safe(self, what: str, fn):
        try:
            return fn()
        except Exception as e:  # noqa: BLE001 - tracing must never break a request
            self.sink_errors.append(f"{what}: {type(e).__name__}")
            if what in ("open", "close"):
                self.cloud = None                      # stop using a broken cloud sink for this run
            return None

    def flush(self) -> None:
        if self.cloud:
            self._safe("flush", self.cloud.flush)


def current_span() -> Span | None:
    return _current.get()


def event(event_name: str, /, **attrs) -> None:
    """Add an event to whatever span is open (no-op outside a span)."""
    span = _current.get()
    if span is not None:
        span.event(event_name, **attrs)


# --- the process-wide tracer ------------------------------------------------------------------------
_tracer: Tracer | None = None


def default_trace_file() -> Path:
    from capstone.settings import get_settings
    return get_settings().runtime_root / "traces" / "traces.jsonl"


def get_tracer() -> Tracer:
    """Local JSONL only, unless configure_tracing(cloud=True) was called by a runner."""
    global _tracer
    if _tracer is None:
        _tracer = Tracer(local=LocalSink(default_trace_file()))
    return _tracer


def configure_tracing(*, cloud: bool = False, path: Path | None = None, cloud_sink: LangfuseSink | None = None) -> Tracer:
    """Called by entry points (runner). Returns the tracer; check `tracer.cloud` to see if Langfuse is on."""
    global _tracer
    sink = None
    if cloud:
        sink = cloud_sink or LangfuseSink()
        if not sink.check():
            sink = None                                # bad keys / offline: keep the local file only
    _tracer = Tracer(local=LocalSink(path or default_trace_file()), cloud=sink)
    return _tracer


def reset_tracing() -> None:
    global _tracer
    _tracer = None


def read_traces(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
