"""The CareFlow FastAPI application. M8.

    .venv/bin/uvicorn capstone.api.main:app --app-dir src --host 127.0.0.1 --port 8000

`create_app(services)` takes the parts the endpoints need, so tests pass fakes (no LLM, no
network) and the real app builds them once at start-up: the LLM client (Claude first, M7 chain),
the M4 policy assistant on the real index, the M6 workflow with durable SQLite checkpoints, the
staff store, and the API key from .env. Langfuse tracing is switched on when its keys work
(M7: otherwise the local trace file only). Served on localhost only (phase 2: no public tunnel).
M8 #6: per-key rate limit (429 + Retry-After) and a body-size limit (413), configs/app.yaml api:.
"""
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from capstone.api.limits import ApiLimits, RateLimiter
from capstone.api.routes import open_router, router
from capstone.settings import get_app_config


@dataclass
class Services:
    graph: Any
    store: Any                                    # StateStore: flags, bookings
    policy: Any                                   # .ask(question) -> PolicyAnswer
    gate_llm: Any                                 # input gate classifier
    safety_llm: Any                               # output gate reviewer
    records: dict                                 # record_id -> intake record (raw text stays here)
    api_key: str | None
    paused: Callable[[], list[dict]]              # review payloads waiting for a clinician
    now: Callable[[], datetime] | None = None
    close: Callable[[], None] = field(default=lambda: None)
    limits: ApiLimits = field(default_factory=ApiLimits)      # M8 #6


def build_services(workdir: Path | None = None) -> Services:
    """The real parts, built once when the server starts. `workdir` keeps the stores, checkpoints,
    memory and sessions of an evaluation run apart from the real runtime/ (runner m8-eval / m8-demo)."""
    from capstone.agents.rag_agent import PolicyAgent
    from capstone.llm.client import LLMClient
    from capstone.memory.long_term import PatientMemory
    from capstone.memory.session import SessionStore
    from capstone.observability.tracing import configure_tracing
    from capstone.settings import get_settings
    from capstone.tools.repository import StateStore, default_state
    from capstone.workflows.checkpoints import Checkpoints, paused_reviews
    from capstone.workflows.graph import build_workflow, load_intake_records

    settings = get_settings()
    configure_tracing(cloud=True, path=workdir / "traces.jsonl" if workdir else None)
    llm = LLMClient(settings)
    policy = PolicyAgent(llm)
    checkpoints = Checkpoints(workdir / "checkpoints" if workdir else None)
    records = dict(load_intake_records())
    store = StateStore(workdir / "state") if workdir else default_state()
    extra = {"memory": PatientMemory(workdir / "memory"), "sessions": SessionStore(workdir / "sessions")} \
        if workdir else {}
    graph = build_workflow(llm, checkpoints.saver, records=records, state_store=store, policy=policy, **extra)
    key = settings.careflow_api_key.get_secret_value() if settings.careflow_api_key else None

    def close() -> None:
        policy.close()
        checkpoints.close()
    return Services(graph=graph, store=store, policy=policy, gate_llm=llm, safety_llm=llm, records=records,
                    api_key=key, paused=lambda: paused_reviews(graph, checkpoints), close=close,
                    limits=get_app_config().api)


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.services = services or build_services()
        try:
            yield
        finally:
            app.state.services.close()

    app = FastAPI(title="CareFlow care-coordination API", version="1.0", lifespan=lifespan)
    limits = services.limits if services is not None else get_app_config().api
    app.state.limiter = RateLimiter(limits.rate_limit_per_minute)

    @app.middleware("http")
    async def body_size_limit(request, call_next):
        size = request.headers.get("content-length")
        if size and size.isdigit() and int(size) > limits.max_body_bytes:
            return JSONResponse({"detail": f"request body over {limits.max_body_bytes} bytes"}, status_code=413)
        return await call_next(request)

    app.include_router(open_router)
    app.include_router(router)
    return app


app = create_app()
