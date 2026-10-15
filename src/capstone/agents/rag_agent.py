"""Policy question answering, end to end: question -> evidence -> grounded answer or refusal. M4.

    PolicyAgent(llm).ask("Does CARD-ECHO need pre-authorisation on MERIDIAN-SILVER?")

1. retrieve: hybrid search by default (dense + keyword, RRF; the measured winner in M4 step 6),
   top 5 passages with their source IDs;
   + M8 fix C1: when the question names a plan, a service and a cost, the co-pay calculator's
   exact result is added as one more source (rag/cost_facts.py; the LLM may only quote it);
2. answer + checks: rag/answer.py (LLM answer, code gates, LLM grounding check, fail safe).
If retrieval itself fails (index missing or built from other passages), the agent returns the
fixed "unavailable" message instead of crashing; it never answers without evidence.
Standalone for now; M6 turns it into the Insurance/Policy RAG specialist agent.
"""
import logging
import time
from typing import Self

from capstone.rag.answer import TEMPLATES, AnswerLLM, GroundingReport, PolicyAnswer, answer_question
from capstone.rag.cost_facts import with_cost_facts
from capstone.rag.retrieval import Mode, PolicyRetriever
from capstone.tools.repository import SeedTables, default_seed

log = logging.getLogger(__name__)
MAX_QUESTION_WORDS = 200


class PolicyAgent:
    def __init__(self, llm: AnswerLLM, retriever=None, seed: SeedTables | None = None):
        self.llm = llm
        self.seed = seed
        self._own = retriever is None
        self.retriever = retriever
        self._error: str | None = None
        if self.retriever is None:
            try:
                self.retriever = PolicyRetriever()
            except Exception as e:  # noqa: BLE001 - no usable index -> every question gets the safe message
                self._error = f"retrieval unavailable: {type(e).__name__}: {e}"

    def ask(self, question: str, mode: Mode | None = None) -> PolicyAnswer:
        question = (question or "").strip()
        if not question:
            raise ValueError("empty question")
        if len(question.split()) > MAX_QUESTION_WORDS:
            return PolicyAnswer(question=question[:2000], route="human_review",
                                answer="The question is too long for the policy assistant; please route it to staff.")
        start = time.perf_counter()
        if self._error is not None:
            return self._unavailable(question, self._error, start)
        try:
            evidence = self.retriever.retrieve(question, mode)
        except Exception as e:  # noqa: BLE001 - search failure -> safe message, never an unsupported answer
            return self._unavailable(question, f"retrieval failed: {type(e).__name__}", start)
        try:
            evidence = with_cost_facts(question, evidence, self.seed or default_seed())
        except Exception as e:  # noqa: BLE001 - calculator problem -> answer from the policies alone
            log.warning("cost facts skipped: %s", type(e).__name__)
        result = answer_question(question, evidence, self.llm)
        result.latency_ms = int((time.perf_counter() - start) * 1000)
        return result

    @staticmethod
    def _unavailable(question: str, error: str, start: float) -> PolicyAnswer:
        return PolicyAnswer(question=question, route="human_review", answer=TEMPLATES["unavailable"],
                            grounding=GroundingReport(checker_error=error),
                            latency_ms=int((time.perf_counter() - start) * 1000))

    def close(self) -> None:
        if self._own and self.retriever is not None:
            self.retriever.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
