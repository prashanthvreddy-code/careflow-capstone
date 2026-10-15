"""Shared setup for the milestone demo scripts.

Makes `src/` and the project root importable, so a script runs as `python milestones/run_mN.py`
from a fresh clone without installing the package. Output shows record IDs and patient_ref only
(no names, dates of birth, member numbers or raw message text).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))


def header(title: str, note: str = "") -> None:
    print(f"\n=== {title} ===")
    if note:
        print(note)


def live_llm():
    """The real LLM client (primary model with fallback), configured from .env."""
    from capstone.llm.client import LLMClient
    from capstone.settings import get_settings
    return LLMClient(get_settings())


def cloud_tracing(workdir: Path) -> str:
    """Send this process's trace spans to Langfuse too (if its keys are in .env; otherwise the
    local file only). Spans carry metadata only, no patient text. Flushed when the process exits."""
    import atexit

    from capstone.observability.tracing import configure_tracing
    from capstone.settings import get_settings

    get_settings()                     # loads .env into the environment, where the Langfuse client reads its keys
    tracer = configure_tracing(cloud=True, path=workdir / "traces.jsonl")
    atexit.register(tracer.flush)
    return "Langfuse + local file" if tracer.cloud else "local file only (no Langfuse keys)"
