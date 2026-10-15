"""Primary-care redirect, prepared by code and sent only after staff approve it. Improvement #3.

referral-policy 3.2.3: a self-referral to a specialty that needs primary care sign-off (3.2.2,
every specialty except Dermatology here) is redirected to the patient's primary care physician.
The M5 decision stays: CareFlow never rejects on its own. When the sign-off evidence is missing it
PREPARES the fixed redirect message (code-written, no identity, no clinical content); a named staff
member sends it with one API call (POST /requests/{id}/pcp-redirect). One redirect per request:
preparing again returns the same one, sending twice is refused.
Stored append-only in the state store ("redirects"): a 'prepared' line, later a 'sent' line.
"""
from datetime import UTC, datetime

from capstone.tools.repository import StateStore

KIND = "redirects"


class NoRedirect(LookupError):
    """No redirect was prepared for this request."""


class AlreadySent(RuntimeError):
    """The redirect for this request was already sent."""


def redirect_message(specialty: str) -> str:
    return (f"Thank you for your request for a {specialty} appointment. Under our referral policy "
            f"(section 3.2), {specialty} needs a referral from your primary care physician first. "
            f"Please contact your Meridian Health Partners primary care physician for an initial "
            f"assessment; they can refer you to {specialty} if needed. - Central Intake Unit")


def _events(store: StateStore, record_id: str) -> list[dict]:
    return [e for e in store.read(KIND) if e["record_id"] == record_id]


def redirect_status(store: StateStore, record_id: str) -> str | None:
    events = _events(store, record_id)
    return events[-1]["status"] if events else None


def prepare_pcp_redirect(store: StateStore, record_id: str, patient_ref: str, specialty: str) -> dict:
    events = _events(store, record_id)
    if events:
        return events[0]
    return store.append(KIND, {"record_id": record_id, "patient_ref": patient_ref, "specialty": specialty,
                               "status": "prepared", "message": redirect_message(specialty),
                               "at": datetime.now(UTC).isoformat(timespec="seconds")})


def send_pcp_redirect(store: StateStore, record_id: str, staff_id: str) -> dict:
    staff_id = (staff_id or "").strip()
    if not staff_id:
        raise ValueError("a named staff member must approve the redirect")
    events = _events(store, record_id)
    if not events:
        raise NoRedirect(record_id)
    if events[-1]["status"] == "sent":
        raise AlreadySent(record_id)
    return store.append(KIND, events[0] | {"status": "sent", "sent_by": staff_id,
                                           "at": datetime.now(UTC).isoformat(timespec="seconds")})


def pending_redirects(store: StateStore) -> list[dict]:
    latest: dict[str, dict] = {}
    for e in store.read(KIND):
        latest[e["record_id"]] = e
    return [e for e in latest.values() if e["status"] == "prepared"]
