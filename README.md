# Applied AI Capstone Starter

**Applied AI Professional Certification Program · IIT Hyderabad**

> **CareFlow submission:** this copy implements the CareFlow problem statement (M1–M8).
> Start with [`MILESTONES.md`](MILESTONES.md): setup, one demo script per milestone, tests and results.

An empty Python source-project scaffold for CareFlow, LexOps, WealthPilot,
ShopSense and PlantGuard. Use one copy per team and implement the selected
problem statement progressively across milestones M1–M8.

## 1. What this project provides

- Importable package directories and empty Python modules with responsibility notes.
- Separate folders for prompts, configuration, data, workflow state, evaluation and tests.
- Data placeholders for all five toolkit domains, including the corpus, intake,
  mock API and evaluation folders.
- A shared milestone map and a domain-specific M1–M8 implementation guide below.
- Git exclusions that retain scaffold folders while excluding data and secrets.

**This is a structure reference, not an implemented solution.** There are no
agents, API endpoints, tool functions, data loaders, evaluation metrics or
working deployment files yet. Python modules contain docstrings only; scripts
perform no operations. No generated records, PDFs, API keys or vendor libraries
are bundled. Do not interpret a successful import as a working milestone.

Assumptions: one project per team, Python 3.12+, FastAPI for
final packaging, and the stack in the supplied problem statements: LiteLLM,
Qdrant, LangGraph, MCP and LangFuse. Cohort dates are intentionally omitted;
follow your current programme schedule and assessment instructions.

## 2. Start using the scaffold

1. Extract the ZIP and open the `applied-ai-capstone-starter/` folder.
2. Select one domain. Keep its data folder; the other four can be removed.
3. Update `configs/app.yaml` and `.env.example` for that domain. `shopsense` is
   only an illustrative default, not an implemented selection mechanism.
4. Create a Python environment from the project root:

```bash
python -m venv .venv
```

Activate it on Linux/macOS:

```bash
source .venv/bin/activate
```

On Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```bat
.venv\Scripts\activate.bat
```

Install the scaffold package:

```bash
python -m pip install -e .
```

This only installs the empty local package; its build backend may require
network access. Add the actual dependencies to `pyproject.toml` as you implement
each milestone. Candidates are `pydantic`, `litellm`, `qdrant-client`,
`langgraph`, the MCP SDK, `langfuse`, `fastapi`, `uvicorn`, configuration/PDF
libraries and `pytest`. Choose compatible versions and commit a lockfile from
your chosen package manager. Do not install every framework before M1.

Copy `.env.example` to `.env` and add necessary secrets locally. Implement
configuration loading in `src/capstone/settings.py`; no automatic loading is
provided. Resolve paths from the project root instead of the current working
directory. Keep model IDs, embedding dimensions and data paths configurable.

## 3. Folder structure and responsibilities

```text
applied-ai-capstone-starter/
    README.md
    pyproject.toml
    .env.example
    .gitignore
    Dockerfile                      # Empty deployment placeholder
    docker-compose.yml              # Empty service placeholder
    configs/                        # App, retrieval, guardrail, eval settings
    data/
        README.md
        careflow/                   # Same generated layout in each domain
        lexops/
        wealthpilot/
        shopsense/
        plantguard/
        external/                   # Approved public datasets and attribution
        processed/                  # Derived chunks and separated intake labels
    src/capstone/
        settings.py
        schemas/                    # intake.py, response.py, state.py
        llm/                        # client.py
        agents/                     # intake, RAG, action and reviewer placeholders
        tools/                      # repository, lookups, calculators, actions
        rag/                        # ingestion, chunking, embeddings, index,
                                    # retrieval, answer
        memory/                     # session.py, long_term.py
        workflows/                  # graph.py, checkpoints.py, approvals.py
        mcp/                        # server.py, client.py
        guardrails/                 # input, output and action checks
        observability/              # tracing.py, logging.py
        reliability/                # retries, fallbacks, circuit breaker
        api/                        # main.py, routes.py
    prompts/                        # Versioned agent instructions
    scripts/                        # Data prep, indexing, state seed, eval entrypoints
    evaluation/                     # runner.py, metrics.py, datasets/, reports/
    tests/                          # unit/, integration/, acceptance/, fixtures/
    runtime/                        # memory, checkpoints, mock_api, outputs, logs
    docs/                           # Architecture, sources, milestones, eval, demo
    notebooks/                      # Optional experiments
```

Use `agents/` for agent behavior, `tools/` for deterministic business operations,
and `workflows/` for orchestration. An agent should call a calculator rather than
ask an LLM to compute a financial ratio or refund amount. MCP exposes the same
tool functions used locally; it should not duplicate business rules.

Start with only the files needed for the next milestone. Add or rename agent
modules for your required specialist roles. The four generic placeholders do
not replace the mandatory domain-specific team in the problem statement.
Keep reusable implementation in `src/capstone/`; notebooks and scripts call it.
Add Python package initializer files when you create new package directories.

`data/` holds inputs, `runtime/` holds mutable application state, and
`evaluation/reports/` holds measured outputs. A Qdrant index, persistent memory
and LangGraph checkpoints serve different purposes and need separate lifecycle
and isolation rules. Configure persistent services/volumes for deployment.

## 4. Generate or import capstone data

Source: https://github.com/AI-Launchpad-Dev/data-toolkit/

| Problem statement | Domain flag | Mock table names (each CSV and JSON) |
|---|---|---|
| CareFlow | `careflow` | plans, patients, appointments, referrals |
| LexOps | `lexops` | counterparties, contracts, signature_requests |
| WealthPilot | `wealthpilot` | applicants, bureau_reports, bank_statements, past_decisions |
| ShopSense | `shopsense` | customers, products, orders, shipments, refunds |
| PlantGuard | `plantguard` | assets, telemetry, work_orders, inventory |

Clone the toolkit alongside this project, not inside its source package:

```bash
git clone https://github.com/AI-Launchpad-Dev/data-toolkit.git
cd data-toolkit/capstone-data-toolkit
```

Follow the toolkit README to create its environment, install its requirements,
and configure its `.env` beside `generate.py`. Toolkit credentials/settings and
application credentials/settings are independent. Use a currently supported
provider model; do not assume a shipped default is still available.

Run these commands **inside the toolkit's `capstone-data-toolkit/` folder**.
For Linux/macOS, replace the example absolute path with your extracted project:

```bash
python generate.py --domain shopsense --out /absolute/path/applied-ai-capstone-starter/data --dry-run
python generate.py --domain shopsense --out /absolute/path/applied-ai-capstone-starter/data
```

For Windows (PowerShell or Command Prompt), for example:

```text
python generate.py --domain shopsense --out "C:\Projects\applied-ai-capstone-starter\data" --dry-run
python generate.py --domain shopsense --out "C:\Projects\applied-ai-capstone-starter\data"
```

Replace the domain flag as needed. `--out` points to the parent `data/` folder;
the generator adds the domain itself. Pointing it to `data/shopsense/` would
create `data/shopsense/shopsense/`.

To begin M2 without any LLM/API key:

```bash
python generate.py --domain shopsense --only tables --out /absolute/path/applied-ai-capstone-starter/data
```

For existing data, copy the whole toolkit `data/<domain>/` folder into this
project's `data/`. Preserve filenames and relative paths. Record the toolkit
commit/version and generation command in `docs/data_sources.md`.

Expected layout after a completed generation (these files are NOT in this ZIP):

```text
data/<domain>/
    manifest.json
    corpus/
        index.json
        markdown/<document-slug>.md
        pdf/<document-slug>.pdf
    intake/records.jsonl
    mock_api/<table-name>.csv
    mock_api/<table-name>.json
    eval/golden_set.json
```

| Asset | Use | Milestone |
|---|---|---|
| Intake records | Parse raw requests; compare extraction with ground truth | M1 |
| Mock tables | Back deterministic lookups and simulated actions; later MCP | M2, M6 |
| Corpus and index | Build retrieval and preserve citation metadata | M3, M4 |
| Golden set | Score answers, citations, expected routes and safety | M4–M8 |
| Manifest | Record provenance and generation settings | Throughout |

Validate actual files and counts; a manifest alone does not prove every
requested asset was produced. Check missing/blocked documents, record shortfalls,
cross-table IDs, numerical consistency and category coverage. Verify golden
expected answers against the generated corpus and mock tables before scoring.

The toolkit's public source references are not downloaded public datasets.
Where required by your statement, add approved datasets separately: for example,
CUAD contracts for LexOps and the specified public sensor dataset for PlantGuard.
Record attribution/licences in `data/external/README.md` and `docs/data_sources.md`.
Use synthetic operational identities; do not import real personal data.

Preserve the generated source folder. Choose CSV or JSON as the table loader
format and seed a mutable copy into `runtime/mock_api/`. For processed files,
use `data/processed/<domain>/documents.jsonl`, `chunks.jsonl`,
`intake_inputs.jsonl` and `intake_labels.jsonl` when you implement preparation.
Markdown/PDF pairs describe the same documents: deduplicate by document slug or
use separate experimental indexes. Never count them as independent evidence.

## 5. Common milestone implementation and acceptance map

| Milestone | Main files/folders | Minimum evidence to demonstrate |
|---|---|---|
| M1: Provider-agnostic intake | `settings.py`, `llm/client.py`, `schemas/intake.py`, intake agent/prompt | Raw request becomes a validated typed object; missing/malformed fields handled; model/provider selected through config; ground truth excluded from model input |
| M2: Tool-enabled single agent | `agents/`, `tools/`, `data/<domain>/mock_api/` | Single agent calls real deterministic mock-backed tools; calculations and tool errors checked; side effects constrained |
| M3: Persistent memory + semantic index | `memory/`, `rag/ingestion.py`, `chunking.py`, `embeddings.py`, `index.py` | Memory survives a restart and is isolated by entity/session; Qdrant retrieves relevant documents with source IDs |
| M4: Production RAG + baseline | `rag/retrieval.py`, `answer.py`, `evaluation/`, retrieval config | Hybrid search/reranking, grounded citations, abstention and PDF ingestion tested; measured baseline recorded |
| M5: Workflow + checkpointing | `schemas/state.py`, `workflows/` | LangGraph conditional routing works; human interrupt persists; approved resume does not repeat an action |
| M6: Multi-agent + MCP | Domain specialist `agents/`, `mcp/`, graph | Required team coordinates with bounded handoffs; MCP tools call the same business logic; reviewer can reject/escalate |
| M7: Observability + reliability | `observability/`, `reliability/`, integration tests | Trace an entire request; simulate timeout/unavailable service; demonstrate bounded retries, circuit breaker and safe fallback |
| M8: Evaluation + guardrails + deployment | `evaluation/`, `guardrails/`, `api/`, deployment files, `docs/` | Run final evaluation (20 cases per statement); demonstrate adversarial controls; serve FastAPI; reproduce setup and demo |

Apply input and action checks from their first use. M8 completes and verifies
guardrails; it is not the first time unsafe actions are constrained.

## 6. How each problem statement fits M1–M8

### 6.1 CareFlow — AI Care Coordination Assistant

Entity scope: synthetic patient + session. Corpus: clinic policies, insurance and
specialty handbooks. Clinical requests require human escalation.

| Milestone | Domain-specific implementation |
|---|---|
| M1 | Define `PatientRequest` with symptoms, urgency and insurance ID; parse messages/referral forms and identify missing information. |
| M2 | Implement mock appointment/EHR and insurance-eligibility lookups, co-pay calculation and a flag-for-human tool. Seed plans/patients/appointments/referrals. Extend mocks when extra fields are needed. |
| M3 | Store isolated patient interaction memory; index policy/insurance documents in Qdrant. |
| M4 | Retrieve correct specialty/coverage evidence with hybrid search/reranking; measure coverage grounding and refuse unsupported coverage claims. |
| M5 | Intake → eligibility check → referral routing; persist a human checkpoint before clinical escalation. |
| M6 | Add Intake, Insurance/Policy RAG, Referral Tracking and Clinical Safety Reviewer agents; expose EHR/scheduling mocks over MCP. |
| M7 | Trace the flow and inject EHR/insurance failures; use bounded retries, circuit breaker and human fallback. |
| M8 | Evaluate 20 cases including diagnosis/dosing bait; block clinical advice, verify escalation, and package the FastAPI service. |

Suggested additions: `agents/insurance_agent.py`, `referral_agent.py`,
`clinical_safety_reviewer.py`. Acceptance focus: no clinical advice, correct
policy citations, patient-memory isolation and no duplicated scheduling.

### 6.2 LexOps — Contract Intelligence & Compliance Copilot

Entity scope: counterparty + contract + session. Corpus: generated clause
playbook and approved contract sources. Recommendations require legal review
where the statement specifies risk/exception conditions.

| Milestone | Domain-specific implementation |
|---|---|
| M1 | Define `ContractSummary`; extract term, termination, liability cap, indemnification and governing law from contract text. |
| M2 | Implement clause-risk calculator, renewal calendar and memo writing. Keep generated memos under `runtime/outputs/`. |
| M3 | Persist counterparty negotiation memory; index playbook and approved redlines/contracts. |
| M4 | Hybrid retrieval/reranking over playbook and contracts; verify cited clauses and source distinctions. |
| M5 | Extract → compare → draft redline → route standard/non-standard cases; checkpoint and interrupt for required human approval. |
| M6 | Add Extraction, Playbook RAG, Redline Drafting and Legal Reviewer agents; expose contract repository/e-signature mocks over MCP. |
| M7 | Trace document review; handle long-document parsing failures and external API timeouts safely. |
| M8 | Evaluate 20 clause cases; require the specified disclaimer and human review, block approval above the risk threshold, and package FastAPI. |

Suggested additions: `agents/clause_extraction_agent.py`, `redline_agent.py`,
`legal_reviewer.py`. Acceptance focus: traceable playbook citations, non-standard
clause escalation, no enforceability claims and no pressure-driven bypass.
The starter includes no CUAD contracts: obtain the approved subset separately.

### 6.3 WealthPilot — SME Loan Underwriting & Credit Research

Entity scope: synthetic applicant + application + session. Corpus: generated
lending policy plus approved regulatory reference material.

| Milestone | Domain-specific implementation |
|---|---|
| M1 | Define `LoanApplication` and `FinancialSnapshot`; parse and validate financial statements, units and missing fields. |
| M2 | Implement DSCR, interest/currency calculators and bureau/bank-statement mock tools; use internally consistent financials. |
| M3 | Persist applicant interaction history; index lending policy and regulatory documents. |
| M4 | Retrieve credit policy using hybrid search/reranking; evaluate grounding and detect fabricated citations. |
| M5 | Intake → document verification → risk scoring → conditional routing; auto-decline only the clear cases specified, flag borderline cases and route remaining decisions to humans. |
| M6 | Add Analyst, Risk Reviewer and Compliance Reviewer committee roles with a bounded actor-critic loop; expose bureau/bank mocks over MCP. |
| M7 | Trace underwriting; simulate bureau failures and demonstrate retries, fallback sources and circuit breaking. |
| M8 | Evaluate 20 applications including matched bias probes; exclude protected attributes/proxies from decision logic, require sign-off above the configured loan threshold and package FastAPI. |

Suggested additions: `agents/risk_scoring_agent.py`, `analyst_agent.py`,
`risk_reviewer.py`, `compliance_reviewer.py`. Acceptance focus: deterministic
financial ratios, policy-grounded rationale, approval gates and identical
decisions/rationales for matched financial profiles differing only in proxies.

### 6.4 ShopSense — Customer Care & Order Operations

Entity scope: synthetic customer + order + session. Corpus: returns, shipping,
warranty, category addenda and escalation policies.

| Milestone | Domain-specific implementation |
|---|---|
| M1 | Define `SupportTicket` with issue type, order ID, sentiment and urgency; handle missing references and injection-bearing ticket text. |
| M2 | Implement order lookup, refund/replace, shipping tracker and refund calculator; use customers/products/orders/shipments/refunds tables. |
| M3 | Persist customer ticket history/preferences; index returns/shipping/warranty policies. |
| M4 | Hybrid retrieval/reranking selects the right product-category exception; evaluate return-window/refund grounding. |
| M5 | Triage → policy check → risk/approval routing → action execution; persist interrupt for high refunds or required sentiment escalation before executing the action. |
| M6 | Add Triage, Policy RAG, Order-Actions and Escalation Reviewer agents; expose order/refund/shipping MCP tools reusable across channels. |
| M7 | Trace ticket resolution; inject order-system failures and demonstrate retries, circuit breaker and human-queue fallback. |
| M8 | Evaluate 20 tickets including angry, ambiguous and injection cases; enforce refund limits in tools, block fabricated policy and serve a FastAPI demo chat endpoint. |

Suggested additions: `agents/triage_agent.py`, `order_actions_agent.py`,
`escalation_reviewer.py`. Acceptance focus: no over-cap refund without approval,
no cross-customer order access and no duplicate refund after retry/resume.

### 6.5 PlantGuard — Maintenance & Supply Chain Operations

Entity scope: asset + maintenance event + session. Corpus: generated manuals,
SOPs, lockout/tagout and procurement policy. Public sensor datasets are separate
from the generated prose alerts and mock operational tables.

| Milestone | Domain-specific implementation |
|---|---|
| M1 | Define `MaintenanceEvent`; reconcile precise sensor alerts with messy logs, units, impossible readings and contradictions. |
| M2 | Implement sensor-history lookup, downtime-cost calculator, inventory lookup and technician scheduling. |
| M3 | Persist asset maintenance history; index equipment manuals and SOPs with asset-specific metadata. |
| M4 | Hybrid retrieval/reranking finds the correct asset manual; evaluate grounded steps and refuse unsupported repair instructions. |
| M5 | Alert intake → root-cause lookup → recommendation → risk routing; auto-log minor issues and interrupt safety-critical recommendations for human approval. |
| M6 | Add Log Intake, Manual RAG, Maintenance Recommendation, Procurement and Safety Reviewer agents; expose inventory/ERP mocks over MCP. |
| M7 | Trace the pipeline; inject sensor-feed/ERP failures; use safe fallback thresholds, bounded retries and circuit breaker. |
| M8 | Evaluate 20 ambiguous/adversarial maintenance scenarios; prevent safety-critical execution without sign-off and package FastAPI with a status-dashboard endpoint. |

Suggested additions: `agents/maintenance_agent.py`, `procurement_agent.py`,
`safety_reviewer.py`. Acceptance focus: correct asset-specific evidence,
impossible-reading handling, approval enforcement and no duplicate procurement.
Do not treat toolkit telemetry as a substitute for the public sensor dataset
required by the statement; do not generate sensor time series with an LLM.

## 7. Evaluation, safety and meaningful tests

- Separate `ground_truth` from intake before passing a request to a model.
- Keep golden `expected`, `must_cite`, `must_not_contain` and `expected_route`
  fields inside the evaluator; only permitted request inputs reach the system.
- Do not index golden cases/answers or memory records into the policy corpus.
- Use extra development cases in `evaluation/datasets/`; freeze the final
  golden set before final scoring. Disclose any cases used for tuning.
- Score extraction, retrieval, citation grounding, refusal and route accuracy
  separately. Report safety failures, latency, cost and domain-specific checks;
  a single overall score can hide critical failures.
- Trace data provenance and configurations for each evaluation run. Store full
  local results in `evaluation/reports/` and a reviewed summary in
  `docs/evaluation_report.md` (tracked in Git).
- `tests/unit/`: deterministic calculations, schemas, thresholds and validators.
- `tests/integration/`: tool data access, retrieval, persistence and MCP behavior.
- `tests/acceptance/`: end-to-end citations, human approval/resume, entity
  isolation, injection refusal, API failures and idempotent actions.
- Use synthetic fixtures/fake provider responses for offline checks. Keep
  live model/service checks explicit; no passing test suite is included.

Add project commands only after implementing their entrypoints. Example future
commands are `python scripts/prepare_data.py`, `python scripts/build_index.py`,
`python scripts/run_evaluation.py`, `python -m pytest` and
`python -m uvicorn capstone.api.main:app --reload`. **They are not operational
in the empty starter.** In particular there is no `app` object yet.

## 8. Commit and share as a Git reference repository

From the extracted project root, if creating a new repository:

```bash
git init
git add .
git status
git commit -m "Add Applied AI capstone project scaffold"
git branch -M main
```

Create an empty remote repository in your preferred account. Substitute its
actual URL in the next command (the placeholder below is not a real URL):

```text
git remote add origin YOUR_REPOSITORY_URL
git push -u origin main
```

For an existing repository, copy the scaffold into it and use its established
branch/workflow; do not reinitialize it or replace an existing remote.

Participants can clone the reference repository or use GitHub's template
repository feature if the maintainer enables it. Each team should maintain its
own implementation and README. Keep this reference clearly labelled as empty.

`.gitkeep` files preserve empty directories through Git; they are placeholders,
not data assets. The ignore rules exclude generated data, mutable state,
evaluation output and secrets while retaining folder markers and README files.
Check `git status` before every commit. Share datasets through the approved
programme channel or generation instructions; Git cloning alone does not fetch
the generated data. An LLM seed does not guarantee byte-identical regeneration:
retain the chosen data snapshot and checksums for reproducible evaluation.

## 9. Final submission checklist

- [ ] Selected domain and mandatory specialist roles implemented.
- [ ] README rewritten with exact working setup, index, start, eval and reset commands.
- [ ] Dependency versions locked; model, embedding and service configuration documented.
- [ ] All four data assets present and validated; provenance/licences recorded.
- [ ] M1–M8 evidence recorded in `docs/milestone_checklist.md`.
- [ ] PDF ingestion and source-grounded answers demonstrated.
- [ ] Persistent entity-scoped memory and approval/checkpoint resume demonstrated.
- [ ] Mock actions obey authorization limits and are safe to retry.
- [ ] MCP integration, traces and injected-service-failure handling demonstrated.
- [ ] Final 20-case evaluation includes domain safety/adversarial cases and failure analysis.
- [ ] FastAPI and deployment files implemented and verified; persistent volumes configured.
- [ ] No secrets, real personal data or golden-label leakage.

## 10. References and scope

- Programme source: supplied **Capstone Problem Statements** document (five
  options, eight milestones). Follow the current teaching/assessment rubric if
  it adds requirements.
- Toolkit overview: https://github.com/AI-Launchpad-Dev/data-toolkit/blob/main/README.md
- CLI/output behavior: https://github.com/AI-Launchpad-Dev/data-toolkit/blob/main/capstone-data-toolkit/generate.py
- Data asset implementation: https://github.com/AI-Launchpad-Dev/data-toolkit/blob/main/capstone-data-toolkit/datagen/domains/base.py

This layout is a recommendation, not a grading requirement by itself. Students
may combine or add files if responsibilities remain clear and milestone
behavior is verified. No upstream toolkit code or datasets are redistributed.
