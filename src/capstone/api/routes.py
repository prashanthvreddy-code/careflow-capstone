"""The CareFlow HTTP endpoints. M8.

Decided in M8 phase 2 ("full API + API key, local"):
    GET  /health                       open; no LLM call, never rate-limited
    POST /policy/ask                   a staff policy question through both guardrails
    POST /requests                     start a patient request through the workflow
    GET  /requests/{record_id}         its status and result
    POST /requests/{record_id}/review  a clinician's answer to a paused request
    GET  /reviews/overdue              paused requests past their first-response limit
    GET  /staff/flags                  the staff / clinician queue
    GET  /staff/redirects              primary-care redirects waiting for approval (improvement #3)
    POST /requests/{record_id}/pcp-redirect   a named staff member sends the prepared redirect
- Every endpoint but /health needs the header X-API-Key (constant-time compare), and is rate
  limited per key after the key check (M8 #6: 429 + Retry-After; bodies over the limit get 413). No key set in
  .env -> the API refuses everything (503): it fails closed, never open.
- Responses are built from small response models: no member numbers, no raw message text, no
  drafts, prompts, traces or internal state. Refusals are visible: a blocked question is an
  HTTP 200 with route "refuse" and the fixed reply; bad input is 422 with the reason.
- The raw message of POST /requests is kept only in memory for the intake step (like the
  intake records file); the workflow state stores the masked request, as before (M5).
"""
import hmac
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field

from capstone.agents.policy_service import InvalidQuestion, ask_guarded
from capstone.schemas.intake import Channel
from capstone.schemas.state import mask_numbers
from capstone.tools.actions import flags_for_record
from capstone.tools.redirects import AlreadySent, NoRedirect, pending_redirects, send_pcp_redirect
from capstone.workflows.approvals import InvalidAnswerError, overdue_reviews
from capstone.workflows.checkpoints import thread_config
from capstone.workflows.graph import (
    NotWaitingError,
    WorkflowResult,
    get_result,
    resume_request,
    start_request,
)

RECORD_ID = r"^CAREFLOW-\d{5}$"
PATIENT_REF = r"^MHP-P-\d{5}$"


def services(request: Request):
    return request.app.state.services


def require_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    expected = services(request).api_key
    if not expected:
        raise HTTPException(503, "the API key is not configured on the server")
    if not x_api_key or not hmac.compare_digest(x_api_key.encode(), expected.encode()):
        raise HTTPException(401, "missing or wrong X-API-Key")
    wait = request.app.state.limiter.check(expected)      # M8 #6: counted only after a valid key
    if wait is not None:
        raise HTTPException(429, "too many requests for this API key", headers={"Retry-After": str(wait)})


open_router = APIRouter()
router = APIRouter(dependencies=[Depends(require_key)])


# --- models (what callers send and get back) ------------------------------------------------------------
class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    question_id: str | None = Field(default=None, pattern=r"^[A-Za-z0-9._-]{3,40}$")


class CitationOut(BaseModel):
    doc_title: str
    section: str
    chunk_id: str


class AskReply(BaseModel):
    question_id: str
    route: str                        # answered / refuse / human_review
    text: str
    citations: list[CitationOut]
    blocked_by: str | None            # input / output / None
    emergency_line: bool
    flag_id: str | None


class StartBody(BaseModel):
    record_id: str = Field(pattern=RECORD_ID)
    patient_ref: str = Field(pattern=PATIENT_REF)
    channel: Channel = "web_form"
    message: str | None = Field(default=None, min_length=1, max_length=4000)   # None: use the stored intake record


class ReviewBody(BaseModel):
    decision: str
    reviewer_id: str
    note: str = ""
    corrected_urgency: str | None = None


class RequestView(BaseModel):
    record_id: str
    patient_ref: str
    status: str
    waiting_for_review: bool
    route: str | None
    urgency: str | None
    specialty: str | None
    flags: list[str]
    booked: bool
    booking_starts_at: str | None
    referral_due_on: str | None
    unavailable: str | None           # M7: which step's EHR / insurance call failed, and why


class FlagOut(BaseModel):
    flag_id: str
    record_id: str
    reason: str
    note: str
    created_at: str


def view(r: WorkflowResult, request_fields: dict | None) -> RequestView:
    req = request_fields or {}
    booking = r.booking or {}
    failure = r.dependency_failure or {}
    return RequestView(
        record_id=r.record_id, patient_ref=r.patient_ref, status=r.status, waiting_for_review=r.waiting_for_review,
        route=r.route, urgency=req.get("urgency"), specialty=req.get("specialty"),
        flags=sorted({f["reason"] for f in r.flags}), booked=bool(booking.get("booked")),
        booking_starts_at=(booking.get("booking") or {}).get("starts_at"),
        referral_due_on=r.referral.due_on if r.referral else None,
        unavailable=f"{failure['step']}: {failure['reason']}" if failure else None)


def _request_fields(svc, record_id: str) -> dict | None:
    snap = svc.graph.get_state(thread_config(record_id))
    return (snap.values or {}).get("request")


# --- endpoints ---------------------------------------------------------------------------------------------
@open_router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.post("/policy/ask", response_model=AskReply)
def policy_ask(body: AskBody, request: Request) -> AskReply:
    svc = services(request)
    try:
        r = ask_guarded(body.question, gate_llm=svc.gate_llm, policy=svc.policy, safety_llm=svc.safety_llm,
                        store=svc.store, qid=body.question_id)
    except InvalidQuestion as e:
        raise HTTPException(422, str(e)) from e
    return AskReply(question_id=r.question_id, route=r.route, text=r.text,
                    citations=[CitationOut(**c.model_dump()) for c in r.citations],
                    blocked_by=r.blocked_by, emergency_line=r.emergency_line, flag_id=r.flag_id)


@router.post("/requests", response_model=RequestView, status_code=201)
def start(body: StartBody, request: Request) -> RequestView:
    svc = services(request)
    known = svc.records.get(body.record_id)
    if body.message is not None:
        if known is not None and known.get("patient_ref") != body.patient_ref:
            raise HTTPException(409, "record_id already belongs to another patient")
        svc.records[body.record_id] = {"record_id": body.record_id, "patient_ref": body.patient_ref,
                                       "channel": body.channel, "raw_text": body.message}
    elif known is None:
        raise HTTPException(422, "no message given and no stored intake record with this id")
    elif known.get("patient_ref") != body.patient_ref:
        raise HTTPException(409, "record_id belongs to another patient")
    result = start_request(svc.graph, body.record_id, body.patient_ref)
    return view(result, _request_fields(svc, body.record_id))


@router.get("/requests/{record_id}", response_model=RequestView)
def get_request(record_id: str, request: Request) -> RequestView:
    svc = services(request)
    result = get_result(svc.graph, record_id)
    if result is None:
        raise HTTPException(404, "unknown request")
    return view(result, _request_fields(svc, record_id))


@router.post("/requests/{record_id}/review", response_model=RequestView)
def review(record_id: str, body: ReviewBody, request: Request) -> RequestView:
    svc = services(request)
    try:
        result = resume_request(svc.graph, record_id, body.model_dump(exclude_none=True))
    except InvalidAnswerError as e:
        raise HTTPException(422, str(e)) from e
    except NotWaitingError as e:
        raise HTTPException(409, str(e)) from e
    return view(result, _request_fields(svc, record_id))


@router.get("/reviews/overdue")
def overdue(request: Request) -> list[dict]:
    svc = services(request)
    late = overdue_reviews(svc.paused(), svc.now() if svc.now else datetime.now(UTC))
    return [{"record_id": o.record_id, "patient_ref": o.patient_ref, "urgency": o.urgency, "paused_at": o.paused_at,
             "waited_hours": o.waited_hours, "limit_hours": o.limit_hours} for o in late]


class RedirectBody(BaseModel):
    staff_id: str = Field(min_length=1, max_length=60, pattern=r"^[A-Za-z0-9._-]+$")


class RedirectOut(BaseModel):
    record_id: str
    patient_ref: str
    specialty: str
    status: str                       # prepared / sent
    message: str                      # the fixed, code-written text (no identity, nothing clinical)
    sent_by: str | None = None


@router.get("/staff/redirects", response_model=list[RedirectOut])
def staff_redirects(request: Request) -> list[RedirectOut]:
    """Primary-care redirects waiting for a staff member's approval (improvement #3)."""
    return [RedirectOut(**r) for r in pending_redirects(services(request).store)]


@router.post("/requests/{record_id}/pcp-redirect", response_model=RedirectOut)
def approve_redirect(record_id: str, body: RedirectBody, request: Request) -> RedirectOut:
    """A named staff member sends the prepared redirect: 404 none prepared, 409 already sent."""
    try:
        return RedirectOut(**send_pcp_redirect(services(request).store, record_id, body.staff_id))
    except NoRedirect:
        raise HTTPException(404, "no primary-care redirect was prepared for this request") from None
    except AlreadySent:
        raise HTTPException(409, "the redirect was already sent") from None


@router.get("/staff/flags", response_model=list[FlagOut])
def staff_flags(request: Request, record_id: str | None = None) -> list[FlagOut]:
    svc = services(request)
    rows = flags_for_record(svc.store, record_id) if record_id else [
        f for rid in dict.fromkeys(r["record_id"] for r in svc.store.read("flags"))
        for f in flags_for_record(svc.store, rid)]
    return [FlagOut(flag_id=f.flag_id, record_id=f.record_id, reason=f.reason.value, note=mask_numbers(f.note),
                    created_at=f.created_at) for f in rows]
