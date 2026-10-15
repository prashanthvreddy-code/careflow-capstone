# CareFlow — milestone guide (M1–M8)

CareFlow is an AI care-coordination assistant for a multi-specialty outpatient clinic chain. It
triages patient intake messages, answers policy and insurance questions with citations, tracks
referrals and bookings through a mock EHR, and never gives clinical advice: clinical content is
always routed to a person.

The code is one Python package (`src/capstone/`); each milestone builds on the previous ones.
This guide maps every milestone to its code, a demo script you can run and its measured
result. Design details: `docs/architecture.md`.

## Setup (fresh clone, Python 3.12)

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/pip install -e .
cp .env.example .env          # then fill in your own keys (see below)
.venv/bin/python -m capstone.rag.index          # build the policy index once (downloads a small local embedding model)
```

`.env` needs: `LLM_MODEL` and `LLM_FALLBACK_MODEL` (LiteLLM names, e.g. `gemini/...`, `anthropic/...`)
with the matching provider keys (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`), and `CAREFLOW_API_KEY`
(any value you choose; the API refuses every request without it). Optional: `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`, `LANGFUSE_BASE_URL` for cloud traces (a local trace file is always written),
`EVAL_JUDGE_MODEL` for the evaluation judge.

Data: `data/careflow/` is the course's synthetic, de-identified toolkit data (policy corpus,
200 intake messages, mock EHR tables, golden set). All paths are relative to the project root.

## Demo scripts

Run from the project root: `.venv/bin/python milestones/run_mN.py`. Output shows record IDs and
`patient_ref` only. Parts marked *offline* run without any API keys (use `--offline` where the script
also has a live part); they still need the `.env` copied from `.env.example`, with the keys left empty.

| | Ask | Main code | Demo script (live LLM calls) | Measured result |
|---|---|---|---|---|
| **M1** | Parse patient messages into a validated `PatientRequest` (symptoms, urgency, insurance ID) with a provider-agnostic LiteLLM client | `schemas/intake.py`, `agents/intake_agent.py`, `llm/client.py`, `prompts/intake.md` | `run_m1.py` — 3 messages (3 calls) | 10/10 valid on the live sample; under-triage 7→1 on all 33 Urgent/Emergent messages |
| **M2** | One agent with tools: mock EHR appointments, insurance eligibility, co-pay calculator, flag for human | `agents/action_agent.py`, `tools/`, `prompts/action.md` | `run_m2.py` — calculator *offline* + 1 record (≈6 calls) | co-pay 30/30 correct; 0 clinical advice |
| **M3** | Session + long-term patient memory; Qdrant index over the policy handbook | `memory/`, `rag/chunking.py`, `rag/embeddings.py`, `rag/index.py` | `run_m3.py` — *offline* (0 calls) | memory survives restart, isolated per patient, member numbers masked; dense MRR 0.825 |
| **M4** | Production RAG over policy PDFs: hybrid search + reranking; grounded answers, no fabricated coverage | `rag/retrieval.py`, `rag/keyword.py`, `rag/rerank.py`, `rag/answer.py`, `rag/cost_facts.py`, `agents/rag_agent.py` | `run_m4.py` — retrieval *offline* + 3 questions (≈6 calls) | hybrid MRR 0.941 (dense 0.825); unseen questions 30/30; 0 unsafe answers |
| **M5** | LangGraph workflow intake → eligibility → routing, with a human checkpoint before clinical escalation | `workflows/graph.py`, `workflows/approvals.py`, `workflows/checkpoints.py`, `workflows/routing.py` | `run_m5.py` — 3 records; the pause is resumed in a separate process (≈4 calls) | 10/10 correct end states; the pause survives a restart; resume never repeats an action |
| **M6** | Full agent team (Intake, Insurance, Referral Tracker, Clinical Safety Reviewer); mock EHR as an MCP server | `agents/`, `mcp/server.py`, `mcp/client.py`, `tools/redirects.py` | `run_m6.py` — MCP *offline* + 2 records (≈10 calls) | routes 10/10 on the live sample; 7 MCP tools; clinical requests always paused |
| **M7** | LangFuse tracing; retries, fallbacks and a circuit breaker against EHR / insurance failures | `observability/`, `reliability/`, `evaluation/m7.py` | `run_m7.py` — *offline* fault scenarios (0 calls) | EHR down: crashes 50→0, every failure visible (0 silent); 0 duplicate bookings |
| **M8** | 20-question golden eval incl. diagnosis bait; guardrails block clinical advice and route to a human; FastAPI service | `guardrails/`, `agents/policy_service.py`, `api/`, `evaluation/m8.py`, `evaluation/m8_random.py` | `run_m8.py` — guardrails + API limits *offline* + 1 answer (≈5 calls) | golden mean 18.7/20 over 3 runs (independent judge); random 29/30; 0 harmful answers, 0 legitimate questions blocked |

## Running the service and the evaluations

```bash
.venv/bin/uvicorn capstone.api.main:app --app-dir src --host 127.0.0.1 --port 8000   # the API: /health, /policy/ask, /requests, /staff/flags
.venv/bin/python -m evaluation.runner m4                 # retrieval: dense vs hybrid vs rerank (no LLM)
.venv/bin/python -m evaluation.runner m7-scenarios       # fault scenarios on 50 records (no LLM)
.venv/bin/python -m evaluation.runner m8-eval --runs 3   # golden set, 3 runs (~270 LLM calls)
```

Only one process at a time can open the local policy index (`runtime/qdrant`), so run the
scripts and evaluations one after another, not in parallel.

## Known limitations

- Over-triage remains (messages rated more urgent than the key); this is the safe direction.
- The policies do not say what a patient owes when the unmet deductible exceeds the list price;
  CareFlow sends these questions to staff.
- The golden set was used for tuning; the unseen and random sets are the fairer measures.
- Rate limits are kept in memory (one process).
