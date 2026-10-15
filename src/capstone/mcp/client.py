"""How the M6 agents reach the EHR / scheduling MCP server. M6.

Decided in M6 phase 2:
- In-process: the workflow talks to the server through an in-memory MCP session. These are real
  MCP messages (initialize, tools/list, tools/call) with no subprocess, so runs and tests are
  fast and repeatable.
- stdio: the same server started as a separate process (`python -m capstone.mcp.server`), the way
  any other MCP client would run it. Used for the demo and for a test that both give the same
  answer.
- Only code calls these tools (no LLM chooses an MCP tool in M6), and code fills in the patient
  reference from the workflow state.
- The MCP SDK is async; the workflow steps are not. `EHRClient` hides that: each call opens a
  session, makes the call, and closes it. A tool error comes back as MCPToolError, never as
  data that could be mistaken for a result.

M7 (phase 2 decisions): every tool call is hardened and traced, in this order:
  span "mcp.<tool>"  ->  retry (3 attempts, transient only)  ->  breaker ("insurance" for
  check_eligibility, "booking" for book_slot (fix F1), "ehr" for the rest)  ->  [seeded fault, scenarios only]  ->  5 s timeout
  ->  the MCP call  ->  strict reply check.
- Strict reply check: every field of the reply model must be PRESENT in the reply. Defaults
  would otherwise turn a reply missing eligibility_status into "not active", a silent wrong
  answer (found in step 3). A bad reply raises MalformedReply (counts against the breaker).
- Whatever cannot be got (retries used up, breaker open, malformed reply) is raised as ONE
  error type, DependencyUnavailable, naming the tool and the reason. A business answer
  ("unknown patient", "slot taken") is still MCPToolError: the EHR worked.
"""
import asyncio
import os
import sys
import threading
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.server.fastmcp import FastMCP
from mcp.shared.memory import create_connected_server_and_client_session
from pydantic import BaseModel, ValidationError

from capstone.mcp.server import BookingRequest, build_server
from capstone.observability.tracing import event, get_tracer
from capstone.reliability.circuit_breaker import CircuitOpenError
from capstone.reliability.config import Reliability, get_reliability
from capstone.reliability.retries import MalformedReply, RetryExhausted, call_with_retry
from capstone.settings import PROJECT_ROOT
from capstone.tools.actions import BookingResult, Flag
from capstone.tools.lookups import AppointmentInfo, EligibilityResult, SlotInfo
from capstone.tools.referrals import ReferralStatus


class MCPToolError(RuntimeError):
    """The server answered a tool call with an error (bad argument, unknown slot, ...)."""


class DependencyUnavailable(RuntimeError):
    """An EHR / insurance call could not be completed: retries used up, breaker open, or bad reply."""

    def __init__(self, tool: str, dependency: str, reason: str, detail: str = ""):
        super().__init__(f"{dependency} unavailable for {tool}: {reason}{' (' + detail + ')' if detail else ''}")
        self.tool, self.dependency, self.reason, self.detail = tool, dependency, reason, detail


DEPENDENCY = {"check_eligibility": "insurance"}          # everything else is the EHR / scheduling system
# M7 fix F1: booking trips its own breaker. Before, 3 failed bookings opened the shared "ehr" breaker,
# so the follow-up schedule check was refused ("status unknown") and other requests' referral and slot
# lookups were refused too. Flags and fault plans still call booking part of the "ehr".
BREAKER = {"check_eligibility": "insurance", "book_slot": "booking"}


def dependency_of(tool: str) -> str:
    return DEPENDENCY.get(tool, "ehr")


def breaker_of(tool: str) -> str:
    return BREAKER.get(tool, "ehr")


def _single(error: BaseException) -> BaseException:
    """The MCP SDK runs calls in task groups; unwrap a one-error ExceptionGroup to the real error."""
    while isinstance(error, BaseExceptionGroup) and len(error.exceptions) == 1:
        error = error.exceptions[0]
    return error


def strict(model: type[BaseModel], data: Any) -> BaseModel:
    """Validate a reply AND require every field to be present (no silent defaults)."""
    if not isinstance(data, dict):
        raise MalformedReply(f"{model.__name__}: expected an object, got {type(data).__name__}")
    missing = sorted(set(model.model_fields) - set(data))
    if missing:
        raise MalformedReply(f"{model.__name__}: missing {missing}")
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise MalformedReply(f"{model.__name__}: {e.error_count()} invalid field(s)") from e


def strict_list(model: type[BaseModel], data: Any) -> list:
    if not isinstance(data, list):
        raise MalformedReply(f"list of {model.__name__}: got {type(data).__name__}")
    return [strict(model, item) for item in data]


def strict_strings(data: Any) -> list[str]:
    if not isinstance(data, list) or not all(isinstance(s, str) for s in data):
        raise MalformedReply("expected a list of strings")
    return list(data)


def stdio_params(extra_env: dict[str, str] | None = None) -> StdioServerParameters:
    """Start the server as `python -m capstone.mcp.server` with the project's src/ on the path."""
    env = {k: v for k, v in os.environ.items() if k in ("PATH", "HOME", "LANG", "RUNTIME_ROOT")}
    env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    env.update(extra_env or {})
    return StdioServerParameters(command=sys.executable, args=["-m", "capstone.mcp.server"],
                                 env=env, cwd=str(PROJECT_ROOT))


def _run(coro):
    """Run a coroutine from plain code, even if an event loop is already running here."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: dict[str, Any] = {}

    def worker() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as e:  # noqa: BLE001 - re-raised in the caller's thread
            box["error"] = e
    t = threading.Thread(target=worker)
    t.start()
    t.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


class EHRClient:
    """Typed, non-async access to the EHR / scheduling MCP server."""

    def __init__(self, server: FastMCP | None = None, *, stdio: StdioServerParameters | None = None,
                 reliability: Reliability | None = None):
        if server is not None and stdio is not None:
            raise ValueError("choose an in-process server or a stdio server, not both")
        self.reliability = reliability or get_reliability()
        self.stdio = stdio
        self.server = None if stdio else (server or build_server())
        self.transport = "stdio" if stdio else "memory"

    async def _session_call(self, op: str, name: str = "", args: dict | None = None):
        async def do(session: ClientSession):
            if op == "list":
                return await session.list_tools()
            return await asyncio.wait_for(session.call_tool(name, args or {}),
                                          timeout=self.reliability.config.mcp_timeout_s)
        if self.stdio:
            async with stdio_client(self.stdio) as (read, write), ClientSession(read, write) as session:
                await session.initialize()
                return await do(session)
        async with create_connected_server_and_client_session(self.server) as session:
            return await do(session)

    def tool_names(self) -> list[str]:
        return sorted(t.name for t in _run(self._session_call("list")).tools)

    def call(self, name: str, parse=None, **args) -> Any:
        """One hardened, traced tool call; returns the (parsed) structured result."""
        rel, dep = self.reliability, dependency_of(name)
        breaker = rel.breakers.get(breaker_of(name))
        with get_tracer().span(f"mcp.{name}", kind="tool", tool=name, dependency=dep,
                               transport=self.transport) as span:
            def once():
                return breaker.call(lambda: self._attempt(name, args, parse))
            try:
                result, attempts = call_with_retry(
                    once, rel.config.mcp_retry, sleep=rel.sleep, rng=rel.rng,
                    on_retry=lambda n, e, wait: event("retry", attempt=n, error=type(e).__name__, wait_s=round(wait, 2)))
            except RetryExhausted as e:
                span.set(attempts=e.attempts, breaker=breaker.state)
                raise DependencyUnavailable(name, dep, "retries_exhausted", type(e.last).__name__) from e
            except CircuitOpenError as e:
                span.set(breaker="open")
                raise DependencyUnavailable(name, dep, "circuit_open") from e
            except MalformedReply as e:
                raise DependencyUnavailable(name, dep, "malformed_reply", str(e)) from e
            span.set(attempts=attempts, breaker=breaker.state)
            return result

    def _attempt(self, name: str, args: dict, parse) -> Any:
        rel = self.reliability
        bad = rel.injector.before_call(name, dependency_of(name), timeout_s=rel.config.mcp_timeout_s) \
            if rel.injector else False
        try:
            result = _run(self._session_call("call", name, args))
        except BaseException as e:  # noqa: BLE001 - re-raised as the real error, unwrapped
            raise _single(e) from None
        if result.isError:
            text = " ".join(getattr(c, "text", "") for c in result.content).strip()
            raise MCPToolError(f"{name}: {text or 'tool error'}")
        data = result.structuredContent
        if data is None:
            raise MalformedReply(f"{name}: no structured result")
        data = data["result"] if set(data) == {"result"} else data
        if bad:
            data = rel.injector.malform(name, data)
        return parse(data) if parse else data

    # --- typed helpers (what the agents use) ------------------------------------------------------
    def check_eligibility(self, patient_ref: str) -> EligibilityResult:
        return self.call("check_eligibility", parse=lambda d: strict(EligibilityResult, d), patient_ref=patient_ref)

    def get_patient_appointments(self, patient_ref: str) -> list[AppointmentInfo]:
        return self.call("get_patient_appointments", parse=lambda d: strict_list(AppointmentInfo, d),
                         patient_ref=patient_ref)

    def get_patient_referrals(self, patient_ref: str) -> list[ReferralStatus]:
        return self.call("get_patient_referrals", parse=lambda d: strict_list(ReferralStatus, d),
                         patient_ref=patient_ref)

    def find_open_slots(self, specialty: str, mode: str = "in_person", limit: int = 3,
                        before: str | None = None) -> list[SlotInfo]:
        return self.call("find_open_slots", parse=lambda d: strict_list(SlotInfo, d),
                         specialty=specialty, mode=mode, limit=limit, before=before)

    def check_booking(self, request: BookingRequest) -> list[str]:
        return self.call("check_booking", parse=strict_strings, request=request.model_dump(mode="json"))

    def book_slot(self, request: BookingRequest) -> BookingResult:
        return self.call("book_slot", parse=lambda d: strict(BookingResult, d), request=request.model_dump(mode="json"))

    def flag_for_human(self, record_id: str, patient_ref: str, reason: str, note: str = "") -> Flag:
        return self.call("flag_for_human", parse=lambda d: strict(Flag, d), record_id=record_id,
                         patient_ref=patient_ref, reason=reason, note=note)


def as_dicts(items: list[BaseModel]) -> list[dict]:
    return [i.model_dump(mode="json") for i in items]
