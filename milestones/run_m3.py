"""M3 - Memory + policy index. No LLM calls.

1. Long-term memory: past interactions from the seed become short, code-built memory items per
   patient (never LLM text, never member numbers); a patient can be forgotten on request.
2. Session memory: one conversation's messages, with member numbers masked as they are stored.
3. Policy index: the policy documents are chunked by section, embedded locally (fastembed, no
   API calls) and stored in an on-disk Qdrant index; dense search plus the M3 retrieval baseline
   on the golden set (document-level MRR and recall@k).
Needs the index: python scripts/build_index.py (once).   Run:  python milestones/run_m3.py
Memory and sessions go to a temporary folder; the index is only read.
"""
import tempfile
from pathlib import Path

try:                                   # as a module (tests) or as a script (python milestones/...)
    from milestones import _common
except ImportError:
    import _common

DEMO_PATIENT = "MHP-P-00236"
QUESTIONS = ("Does a cardiac echo need pre-authorisation on the Meridian Silver plan?",
             "What is the maximum CCTV footage retention period at facility access points?")


def memory(workdir: Path, patient_ref: str = DEMO_PATIENT) -> dict:
    from capstone.memory.long_term import PatientMemory
    from capstone.tools.repository import default_seed

    mem = PatientMemory(workdir / "memory")
    imported = mem.import_seed(default_seed())
    again = mem.import_seed(default_seed())                    # repeat-safe: nothing new
    history = [{"occurred_at": h.occurred_at[:10], "topic": str(h.topic), "outcome": str(h.outcome),
                "status": str(h.status)} for h in mem.history(patient_ref, limit=5)]
    forgotten = mem.forget_patient(patient_ref)
    return {"imported": imported, "reimported": again, "patient_ref": patient_ref, "history": history,
            "forgotten": forgotten, "after_forget": len(mem.history(patient_ref))}


def session(workdir: Path, patient_ref: str = DEMO_PATIENT) -> dict:
    from capstone.memory.session import SessionStore
    from capstone.tools.repository import default_seed

    member = default_seed().get("patients", patient_ref=patient_ref)["member_number"]   # read, never printed
    store = SessionStore(workdir / "sessions")
    s = store.open(patient_ref, "CAREFLOW-00001")
    s.add({"role": "user", "content": f"Please book me, my member number is {member}."})
    s.add({"role": "assistant", "content": "Checking your coverage now."})
    store.close(s)
    saved = store.load(patient_ref, s.session_id)
    return {"messages": len(saved.messages), "redactions": saved.redactions,
            "stored_text": saved.messages[0]["content"], "approx_tokens": saved.approx_tokens}


def index_demo(index, questions: tuple[str, ...] = QUESTIONS) -> dict:
    from evaluation.runner import run_retrieval_eval

    searches = [{"question": q, "top": [(h.doc_title, h.section, round(h.score, 3)) for h in index.search(q, top_k=3)]}
                for q in questions]
    overall = run_retrieval_eval(index)["overall"]
    return {"searches": searches, "baseline": {k: round(v, 3) for k, v in overall.items()}}


def show(mem: dict, ses: dict, idx: dict | None) -> None:
    _common.header("M3 Part 1: long-term memory (temp folder)")
    print(f"  imported from seed {mem['imported']}; second import {mem['reimported']}")
    print(f"  last {len(mem['history'])} items for {mem['patient_ref']}:")
    for h in mem["history"]:
        print(f"    {h['occurred_at']} {h['topic']:14} {h['outcome']:14} {h['status']}")
    print(f"  forget patient -> {mem['forgotten']}; items left {mem['after_forget']}")
    _common.header("M3 Part 2: session memory")
    print(f"  {ses['messages']} messages, {ses['redactions']} member number masked, ~{ses['approx_tokens']} tokens\n"
          f"  stored as: {ses['stored_text']}")
    if idx is None:
        return
    _common.header("M3 Part 3: policy index (dense search, local embeddings)")
    for s in idx["searches"]:
        print(f"  Q: {s['question']}")
        for title, section, score in s["top"]:
            print(f"     {score:.3f}  {title} / {section}")
    print(f"  golden-set baseline: {idx['baseline']}")


if __name__ == "__main__":
    from capstone.rag.index import open_policy_index

    with tempfile.TemporaryDirectory() as tmp, open_policy_index() as index:
        show(memory(Path(tmp)), session(Path(tmp)), index_demo(index))
