# Architecture

CareFlow is an AI care-coordination assistant for a multi-site outpatient clinic (synthetic data:
"Meridian Health Partners"). It turns a patient's message into a structured request, checks
coverage and referrals, books what it may book, answers staff policy questions with citations,
and sends every clinical question to a licensed clinician. **It never gives clinical advice.**

## Request flow (patient messages)

```
POST /requests ─► intake (LLM, M1) ─► eligibility (MCP, M6) ─► route (code)
   route: emergency        ─► fixed 911/ED notice ─► clinical pause (human)
          clinical_review  ─► clinical pause (human, LangGraph interrupt, SQLite checkpoint)
          staff_queue      ─► staff flags
          referral         ─► insurance (co-pay + cited policy answer) ─► tracking (referrals,
                              deadline, booking plan) ─► Clinical Safety Reviewer ─► book ─► record
POST /requests/{id}/review ─► escalate | not_clinical | handled (named reviewer) ─► resume
```

- **Workflow:** LangGraph (`workflows/graph.py`), code router, bounded edges, durable SQLite
  checkpoints; every side effect is idempotent (`guardrails/action_checks.py` lists them all).
- **Agents as steps:** Intake (M1 LLM extraction), Insurance (eligibility over MCP, co-pay
  calculator, one cited policy answer), Referral Tracker (deadlines, policy 3.5 escalation
  levels, earliest slot before the deadline), Clinical Safety Reviewer (code gates + 1 LLM check).
- **EHR / scheduling:** an MCP server (`mcp/server.py`, 7 tools) over the mock tables; the
  workflow is its client. No LLM ever chooses a tool or a side effect.

## Policy questions (staff)

```
POST /policy/ask ─► input gate (code rules from the Scope-of-Practice Standard §3, then 1 LLM
                    classifier) ─► M4 policy assistant (hybrid BM25 + dense retrieval over the 14
                    policy PDFs; cost questions also get the co-pay rules and, when plan + service
                    are named, the co-pay calculator's result as a source; cited JSON answer; code
                    checks + grounding check, one redraft) ─► output gate (M6 reviewer)
                    ─► answer with citations | code-written deflection (§4.3 verbatim, 911 line
                    first on emergency signs) + clinician flag
```

## Schemas

`PatientRequest` (M1), `WorkflowState` (M5: masked request only, never raw text), `PolicyAnswer`
(M4), `GuardedAnswer` (M8), API response models (`api/routes.py`: no member numbers, no raw
text, no drafts or traces).

## Memory

Session memory per request run, long-term patient memory per `patient_ref` (JSONL, written by
code only, source-tagged, `forget_patient`), the policy index (Qdrant embedded on disk,
bge-small-en-v1.5 + an ms-marco MiniLM reranker, both local).

## Human approval

Clinical and Emergent requests pause before any clinical escalation; a named reviewer answers;
the resume never repeats an action. Blocked policy questions become `clinical_question` flags.
A referral without primary-care sign-off gets a prepared redirect (referral-policy 3.2.3) that is
sent only when a named staff member approves it (`POST /requests/{id}/pcp-redirect`).

## API limits

X-API-Key on every endpoint but `/health`; 60 requests a minute per key (429 + Retry-After);
bodies over 16 KB refused (413). In memory, one process.

## Reliability and observability (M7)

Every MCP and LLM call: timeout → capped retries for brief failures → circuit breaker
(insurance / ehr / booking / per LLM model) → strict reply check. A dependency that stays down
sends the request to staff with `dependency_unavailable`; emergency and clinical paths never
wait on it. Every request is one trace (Langfuse + local JSONL), metadata only.

## Service dependencies

| Dependency | Used for | If it fails |
|---|---|---|
| Claude (primary LLM) → Gemini (fallback) via LiteLLM | intake, policy answers, grounding, safety review, input gate | fallback model; then the request or question goes to a person |
| MCP EHR / scheduling server (in-process) | eligibility, referrals, slots, booking | retries + breaker; staff flag |
| Qdrant (embedded, on disk) | policy retrieval | "assistant unavailable" template, staff review |
| Langfuse Cloud | traces | local trace file only; requests unaffected |
