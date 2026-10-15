"""M4 - Hybrid RAG with citations.

Part 1 (no LLM): the same question searched three ways - dense (M3), hybrid (dense + keyword,
fused with reciprocal rank fusion) and hybrid + local re-ranker - plus each mode's MRR on the
golden set. Topic routes in configs/retrieval.yaml pin known sections for some intents.
Part 2 (live): three questions answered by the PolicyAgent. Every answer must cite the passages it
uses; code checks numbers and citations, an LLM fact-check checks every claim against the cited
passages, and a failed draft gets one redraft or is withheld. For cost questions, code runs the
co-pay calculator and adds its result as a source (the LLM never does the arithmetic).
Live: 3 questions, about 1-4 LLM calls each. The third is not covered by any policy, so it is refused.
Run:  python milestones/run_m4.py            (both parts; needs the index)
      python milestones/run_m4.py --offline  (Part 1 only, no keys needed)
"""
import sys

try:                                   # as a module (tests) or as a script (python milestones/...)
    from milestones import _common
except ImportError:
    import _common

SEARCH_Q = "What is the maximum CCTV footage retention period at facility access points?"   # exact term: keyword helps
ASK = (SEARCH_Q,
       "What will a CARD-ECHO cost a patient on the MERIDIAN-SILVER plan?",                  # calculator source
       "What happens if a patient misses an appointment without cancelling?")                # not in the policies

MODES = ("dense", "hybrid", "hybrid_rerank")


def compare_modes(retriever, question: str = SEARCH_Q) -> dict:
    from evaluation.runner import GOLDEN, run_variant_eval

    top = {m: [f"{h.doc_title} / {h.section}" for h in retriever.retrieve(question, m, top_k=3).hits] for m in MODES}
    golden = run_variant_eval(retriever, {"golden": GOLDEN})["results"]["golden"]
    return {"question": question, "top": top, "mrr": {m: round(golden[m]["overall"]["mrr"], 3) for m in MODES}}


def answers(llm, retriever, questions: tuple[str, ...] = ASK) -> list[dict]:
    from capstone.agents.rag_agent import PolicyAgent

    agent = PolicyAgent(llm, retriever=retriever)
    rows = []
    for q in questions:
        a = agent.ask(q)
        rows.append({"question": q, "route": a.route, "withheld": a.withheld, "answer": a.answer,
                     "citations": [f"{c.doc_title} / {c.section}" for c in a.citations],
                     "fact_check_passed": a.grounding.llm_grounded, "redrafted": a.grounding.redrafted,
                     "latency_ms": a.latency_ms})
    return rows


def show_modes(r: dict) -> None:
    _common.header("M4 Part 1: three retrieval modes (no LLM)",
                   f"Q: {r['question']}\n(the app uses hybrid: the re-ranker measured lower on these policies)")
    for m in MODES:
        print(f"  {m:14} golden MRR {r['mrr'][m]}")
        for i, t in enumerate(r["top"][m], 1):
            print(f"     {i}. {t}")


def show_answers(rows: list[dict]) -> None:
    _common.header("M4 Part 2: cited answers (live)")
    for r in rows:
        print(f"  Q: {r['question']}\n  route {r['route']}, withheld {r['withheld']}, fact-check passed "
              f"{r['fact_check_passed']}, redrafted {r['redrafted']}, {r['latency_ms']} ms\n  A: {r['answer']}")
        for c in r["citations"]:
            print(f"     cites {c}")


if __name__ == "__main__":
    from capstone.rag.retrieval import PolicyRetriever

    with PolicyRetriever() as retriever:
        show_modes(compare_modes(retriever))
        if "--offline" not in sys.argv:
            show_answers(answers(_common.live_llm(), retriever))
