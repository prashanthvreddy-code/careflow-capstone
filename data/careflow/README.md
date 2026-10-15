# CareFlow data

The synthetic, de-identified dataset from the course data toolkit. It is all invented: no real
patients. The code reads it from here (paths relative to the project root, see
`src/capstone/settings.py`).

- `manifest.json`: what the toolkit generated
- `corpus/markdown/*.md`, `corpus/pdf/*.pdf`, `corpus/index.json`: the 14 clinic policy documents
- `intake/records.jsonl`: 200 patient intake messages with answer-key labels
- `eval/golden_set.json`: the 20-question golden evaluation set
- `mock_api/*.csv|json`: mock EHR, scheduling and insurance tables (patients, plans, co-pays,
  procedures, appointments, slots, referrals, providers, sites)

Read-only: the application writes its state under `runtime/`, never here.
