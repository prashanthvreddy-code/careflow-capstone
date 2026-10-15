"""Project settings loaded from environment variables / `.env`. M1 onward.

Only *names* of models live here; API keys stay in the environment and are read by LiteLLM itself
(GEMINI_API_KEY, GROQ_API_KEY, ANTHROPIC_API_KEY). Never log or print a key.

M3 adds runtime paths for memory and the retrieval config (configs/retrieval.yaml), which is
validated here so a typo fails at start-up rather than in the middle of indexing.
"""
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from capstone.api.limits import ApiLimits
from capstone.reliability.config import ReliabilityConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RETRIEVAL_CONFIG = PROJECT_ROOT / "configs" / "retrieval.yaml"
EVAL_DIR = PROJECT_ROOT / "data" / "careflow" / "eval"


class Settings(BaseSettings):
    """Typed view of `.env`. Field `llm_model` is read from env var `LLM_MODEL`, and so on."""

    model_config = SettingsConfigDict(env_file=PROJECT_ROOT / ".env", extra="ignore")

    llm_model: str  # required, e.g. "gemini/gemini-2.5-flash" or "groq/..."
    llm_fallback_model: str | None = None  # None = no fallback, e.g. "anthropic/claude-..."
    llm_temperature: float = 0.0  # extraction should be repeatable
    llm_max_tokens: int = 2000  # A1 lesson: always cap output tokens; M8: reasoning tokens count too
    llm_max_retries: int = 2  # repair-loop attempts after the first try
    data_root: Path = PROJECT_ROOT / "data" / "careflow"
    runtime_root: Path = PROJECT_ROOT / "runtime"  # M3: mutable state (memory, index), gitignored
    session_token_budget: int = 8000  # M3: a session over this size is reported, never silently cut
    careflow_api_key: SecretStr | None = None  # M8: X-API-Key for the FastAPI service (None = API refuses)
    eval_judge_model: str | None = None  # M8 #4: the evaluation judge's own model (None = same as llm_model)

    @property
    def memory_dir(self) -> Path:
        """Long-term patient memory: one file per patient_ref."""
        return self.runtime_root / "memory" / "patients"

    @property
    def sessions_dir(self) -> Path:
        """Closed sessions (one request run each), kept for audit."""
        return self.runtime_root / "memory" / "sessions"


@lru_cache
def get_settings() -> Settings:
    """Return one shared Settings instance (cached so `.env` is read once)."""
    # pydantic-settings reads .env into Settings only; LiteLLM looks for API keys in os.environ,
    # so load .env there too. override=False: a key already set in the shell wins.
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    return Settings()


APP_CONFIG = PROJECT_ROOT / "configs" / "app.yaml"


class AppConfig(BaseModel):
    """configs/app.yaml. M6: as_of_date is the date the system treats as 'today'. M7: reliability limits."""

    model_config = ConfigDict(extra="ignore")

    domain: str = "careflow"
    as_of_date: date
    reliability: ReliabilityConfig = ReliabilityConfig()       # M7: timeouts, retries, breakers
    api: ApiLimits = ApiLimits()                                # M8 #6: rate and body-size limits


@lru_cache
def get_app_config(path: Path = APP_CONFIG) -> AppConfig:
    return AppConfig.model_validate(yaml.safe_load(path.read_text()) or {})


# ---------------------------------------------------------------------------------------------
# M3 retrieval config (configs/retrieval.yaml)
# ---------------------------------------------------------------------------------------------
class CorpusConfig(BaseModel):
    source: Literal["markdown", "pdf"] = "markdown"
    markdown_dir: Path
    pdf_dir: Path | None = None


class ChunkingConfig(BaseModel):
    strategy: Literal["section"] = "section"
    max_words: int = Field(300, ge=50, le=2000)


class EmbeddingConfig(BaseModel):
    provider: Literal["fastembed"] = "fastembed"
    model: str
    dimension: int = Field(gt=0)


class IndexConfig(BaseModel):
    backend: Literal["qdrant_local"] = "qdrant_local"
    path: Path
    collection: str
    distance: Literal["cosine"] = "cosine"
    top_k: int = Field(5, ge=1, le=50)


class TopicRoute(BaseModel):
    """M4 fix X1b: a question using one of `terms` always gets the `anchor` passage in its evidence."""

    name: str
    anchor: str                         # a chunk_id, e.g. "copay-schedule#4.2"
    terms: list[str] = Field(min_length=1)


class SearchConfig(BaseModel):
    """M4: how candidates are found and ordered."""

    mode: Literal["dense", "hybrid", "hybrid_rerank"] = "hybrid"
    dense_pool: int = Field(10, ge=1, le=100)
    keyword_pool: int = Field(10, ge=1, le=100)
    rrf_k: int = Field(60, ge=1, le=1000)
    rerank_pool: int = Field(10, ge=1, le=100)
    topic_routes: list[TopicRoute] = []


class RerankerConfig(BaseModel):
    provider: Literal["fastembed"] = "fastembed"
    model: str


class RetrievalConfig(BaseModel):
    corpus: CorpusConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    index: IndexConfig
    retrieval: SearchConfig = SearchConfig()
    reranker: RerankerConfig | None = None

    @model_validator(mode="after")
    def _resolve_and_guard(self) -> "RetrievalConfig":
        """Make paths absolute (from the project root) and refuse to index evaluation data."""
        self.corpus.markdown_dir = _from_root(self.corpus.markdown_dir)
        if self.corpus.pdf_dir:
            self.corpus.pdf_dir = _from_root(self.corpus.pdf_dir)
        self.index.path = _from_root(self.index.path)
        for folder in (self.corpus.markdown_dir, self.corpus.pdf_dir):
            if folder and folder.is_relative_to(EVAL_DIR):
                raise ValueError("a corpus folder points into the evaluation folder; never index eval data")
        if self.corpus.source == "pdf" and not self.corpus.pdf_dir:
            raise ValueError("corpus.source is pdf but corpus.pdf_dir is not set")
        if self.retrieval.mode == "hybrid_rerank" and not self.reranker:
            raise ValueError("retrieval.mode hybrid_rerank needs a reranker section")
        if self.retrieval.rerank_pool < self.index.top_k:
            raise ValueError("retrieval.rerank_pool must be at least index.top_k")
        return self

    @property
    def corpus_dir(self) -> Path:
        """The folder of the copy that is indexed."""
        return self.corpus.pdf_dir if self.corpus.source == "pdf" else self.corpus.markdown_dir


def _from_root(path: Path) -> Path:
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def get_retrieval_config(path: Path = RETRIEVAL_CONFIG) -> RetrievalConfig:
    """Load and validate the retrieval config (not cached, so tests can pass their own file)."""
    with open(path, encoding="utf-8") as fh:
        return RetrievalConfig.model_validate(yaml.safe_load(fh))
