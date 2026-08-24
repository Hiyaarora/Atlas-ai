"""Application configuration.

Every tunable value in Atlas AI is read from the environment exactly once,
here, and validated by Pydantic at import time. The rest of the codebase
imports `settings` and never touches `os.environ` directly.

Why this matters: a typo in an env var name fails loudly at startup instead
of silently at 2am when the code path is finally hit.
"""

from functools import lru_cache
from typing import Annotated, Literal

from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

Environment = Literal["development", "staging", "production"]


class Settings(BaseSettings):
    """Typed, validated view of the process environment."""

    model_config = SettingsConfigDict(
        # The repo root holds the single .env shared by every service. The
        # second entry lets you drop a backend-only .env for one-off overrides;
        # later files win.
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # frontend VITE_* vars share the file; don't choke on them
    )

    # ---- Application -----------------------------------------------------
    app_name: str = "Atlas AI"
    app_env: Environment = "development"
    debug: bool = False
    log_level: str = "INFO"
    log_format: Literal["console", "json"] = "console"

    # ---- API -------------------------------------------------------------
    api_v1_prefix: str = "/api/v1"

    # NoDecode is load-bearing. Without it, pydantic-settings sees a list type,
    # assumes the env value is JSON, and calls json.loads() on it *before* any
    # validator runs - so `CORS_ORIGINS=http://a,http://b` raises a
    # JSONDecodeError at import. NoDecode hands the raw string to the
    # `mode="before"` validator below instead.
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=list)

    # ---- Security --------------------------------------------------------
    secret_key: str = "insecure-development-key"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 30
    bcrypt_rounds: int = 12  # ~250ms per hash; raise as hardware improves

    # Refresh-token cookie. `secure=True` requires HTTPS, so it is off in
    # local development and must be on everywhere else.
    refresh_cookie_name: str = "atlas_refresh"
    refresh_cookie_secure: bool = False
    refresh_cookie_samesite: Literal["lax", "strict", "none"] = "lax"

    # ---- Testing ---------------------------------------------------------
    postgres_test_db: str = "atlas_test"

    # ---- PostgreSQL ------------------------------------------------------
    postgres_user: str = "atlas"
    postgres_password: str = "atlas"
    postgres_db: str = "atlas"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    # ---- LLM -------------------------------------------------------------
    # `echo` is a deterministic local double used by the test suite.
    llm_provider: Literal["gemini", "echo"] = "gemini"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-2.5-flash"

    llm_temperature: float = 0.7

    # Raised from 2048 after measuring truncation. See llm_thinking_budget:
    # the two interact, and the old value was being consumed before the model
    # had finished speaking.
    llm_max_output_tokens: int = 4096

    # Tokens the model may spend on internal reasoning it never shows.
    #
    # gemini-2.5-flash reasons before answering by default, and those tokens
    # are charged against max_output_tokens. Measured on a single ordinary
    # question with the previous 2048 budget:
    #
    #   finish_reason      MAX_TOKENS
    #   thinking tokens    1760      <- invisible
    #   visible tokens      284      <- what the user got
    #
    # 86% of the budget went to reasoning nobody sees, and the answer stopped
    # mid-sentence. Setting 0 disables thinking: the same question then
    # produced 15945 characters instead of 1330.
    #
    # 0 is right for a grounded assistant, whose job is to report what the
    # retrieved passages say rather than to reason its way to new conclusions.
    # Raise it if a task genuinely needs deliberation, and raise
    # llm_max_output_tokens with it or the visible answer loses the room.
    llm_thinking_budget: int = 0

    llm_request_timeout_seconds: float = 60.0

    # Conversations grow without limit; context windows do not. Only the most
    # recent N messages are replayed to the model.
    llm_history_message_limit: int = 20

    # ---- Embeddings ------------------------------------------------------
    # `fake` is deterministic and offline, used by the test suite.
    embedding_provider: Literal["gemini", "fake"] = "gemini"
    # NOTE: text-embedding-004 was retired and now 404s. Verified working.
    gemini_embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    # Chunks per API call. Too small wastes round trips; too large risks the
    # request payload limit and makes one failure cost more work.
    embedding_batch_size: int = 64

    # ---- Chunking --------------------------------------------------------
    # Characters, not tokens. Roughly 1000 chars ~ 250 tokens for English.
    chunk_size: int = 1000
    # Overlap keeps a sentence that straddles a boundary retrievable from
    # both sides. ~20% is the usual starting point.
    chunk_overlap: int = 200

    # ---- Uploads ---------------------------------------------------------
    max_upload_bytes: int = 20 * 1024 * 1024
    storage_dir: str = "storage"
    chroma_dir: str = "storage/chroma"

    # ---- Retrieval -------------------------------------------------------
    retrieval_top_k: int = 6

    # Absolute floor: below this, a hit is noise under any model.
    #
    # Was 0.35, which was a guess and effectively disabled the gate. Calibrated
    # against the 53-query golden set with `python -m app.eval.run --calibrate`:
    #
    #   relevant chunk, answerable     min=0.596  p50=0.684  max=0.804
    #   best chunk, UNANSWERABLE       min=0.520  p50=0.604  max=0.680
    #
    # Note those ranges OVERLAP: no absolute floor separates them, which is why
    # this is only half the gate (see retrieval_min_margin). 0.55 sits ~0.046
    # below the worst observed relevant score — deliberately not the 0.59 the
    # sweep suggested, which was tuned to the exact edge of 53 samples and
    # would reject a relevant chunk scoring 0.58 on any unseen query.
    retrieval_min_score: float = 0.55

    # Distinctiveness floor: how far the best chunk must stand above the
    # median chunk for the SAME query.
    #
    # Absolute similarity carries a large query-dependent baseline — a verbose
    # question is closer to everything than a terse one — so 0.62 means
    # different things for different queries. Subtracting the median removes
    # that baseline: when the corpus holds the answer one chunk stands out;
    # when it does not, everything is uniformly unrelated.
    #
    #   answerable      margin  min=0.017  p50=0.073  max=0.152
    #   UNANSWERABLE    margin  min=0.015  p50=0.044  max=0.071
    #
    # Also overlapping, and also only useful jointly. Set below the worst
    # observed answerable margin for headroom; re-derive with --calibrate
    # after any change to the embedding model or chunk size, both of which
    # move these distributions.
    retrieval_min_margin: float = 0.010

    # A median needs a population. Below this many candidates the margin test
    # is skipped rather than computed from two numbers, because a "median" of
    # two chunks says nothing about whether one stands out.
    retrieval_margin_min_candidates: int = 4

    # Relative cutoff: keep only hits scoring at least this fraction of the
    # best hit's score.
    #
    # Measured with gemini-embedding-001: for the question "how do I stop the
    # primary discarding logs my replica needs", the correct passage scored
    # 0.677 while two entirely unrelated passages scored 0.586 and 0.583. An
    # absolute threshold cannot separate those — anything that excludes 0.586
    # would exclude genuinely relevant passages on an easier question.
    #
    # The *gap* is the signal, not the magnitude. 0.586/0.677 = 0.87, so a
    # 0.90 cutoff drops both. Tuned properly against a labelled set by the
    # evaluation harness; an evidence-based starting point, not a final answer.
    #
    # NOTE: applies to the dense-only path only. The hybrid path gates on each
    # retriever's native score instead — see `retrieval/pipeline.py`.
    retrieval_relative_cutoff: float = 0.90

    # ---- Hybrid retrieval ------------------------------------------------
    # The switch that makes "hybrid vs dense-only baseline" measurable rather
    # than a matter of opinion. The evaluation harness flips it to compare.
    retrieval_hybrid_enabled: bool = True

    # How deep EACH retriever goes before fusion. Larger than retrieval_top_k
    # on purpose: fusion can only reorder what it is given, so consensus
    # between retrievers is invisible unless both lists run past the final cut.
    retrieval_candidate_k: int = 20

    # RRF's rank-smoothing constant. 60 is the value from Cormack et al. (2009)
    # and encodes "agreement between retrievers beats confidence from one".
    retrieval_rrf_k: int = 60

    # Lexical relevance is judged RELATIVE to the best BM25 hit for the same
    # query, because BM25 is unbounded — an absolute floor would mean something
    # different for every query and every corpus size. 0.30 keeps hits within
    # roughly a third of the best match.
    retrieval_lexical_relative_cutoff: float = 0.30

    # ---- Cross-encoder reranking -----------------------------------------
    # DEFAULT OFF, deliberately, and not because the feature is unfinished.
    #
    # Reranking costs ~500ms p50 on CPU for a dozen 1000-character candidates,
    # roughly doubling retrieval latency. The offline probe
    # (`python -m app.eval.rerank_probe`) shows it ranks better than RRF —
    # MRR 0.921 vs 0.885, rank-1 on 39/45 — but that measures the cross-encoder
    # scoring a whole document, not the pipeline it actually sits in.
    #
    # The rule here is "only keep the reranker if it provides measurable
    # improvement", and the pipeline A/B has not run yet: Gemini's
    # 1000-per-day embedding cap was exhausted mid-benchmark. Shipping it on by
    # default before that number exists would be exactly the intuition-driven
    # decision this stage was meant to avoid.
    #
    # Flip to true once `python -m app.eval.run --config rerank` justifies the
    # latency.
    rerank_enabled: bool = False

    # A small MS MARCO cross-encoder: 6 layers, ~90 MB, trained on the task of
    # scoring (query, passage) pairs for relevance. Bigger variants score
    # slightly better and cost proportionally more per query; that trade is
    # measurable with `python -m app.eval.run` once there is a reason to test it.
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # How many fused candidates reach the reranker. Larger than
    # retrieval_top_k: the reranker's value is promoting something retrieval
    # ranked 12th into the final six, which it cannot do if only six arrive.
    # Cost is linear — every candidate is a forward pass — so this is the main
    # latency dial.
    rerank_candidate_k: int = 20

    # Pairs scored per forward batch. Tuned for CPU; larger batches help GPUs
    # far more than they help here.
    rerank_batch_size: int = 16

    # Tokens per (query, passage) pair. Chunks are ~1000 characters, roughly
    # 250 tokens, so 512 leaves comfortable room for the query. Anything longer
    # is truncated by the model rather than erroring.
    rerank_max_length: int = 512

    # BM25 term-frequency saturation. Higher = repetition keeps mattering.
    bm25_k1: float = 1.2
    # BM25 length normalisation. 0 = ignore document length, 1 = full.
    bm25_b: float = 0.75

    # Minimum fraction of a query's idf mass a chunk must match to count as a
    # lexical hit at all. This is the ABSOLUTE floor for BM25 — expressible
    # only because coverage is normalised against the query rather than
    # against the other candidates, unlike the raw score.
    #
    # Without it, "zebra migration across the serengeti" asked of a Postgres
    # manual matches "across", becomes the best lexical hit by default, and
    # passes a purely relative cutoff at ratio 1.0. Caught by
    # test_weak_matches_are_filtered_out.
    bm25_min_coverage: float = 0.30

    # ---- Whole-document requests -----------------------------------------
    # "Summarise this document" is not a retrieval task: it describes an
    # operation over the whole document, not a passage to find. Such requests
    # bypass vector search and load the document itself, up to this budget.
    #
    # ~120k characters is roughly 30k tokens — comfortable for Gemini's
    # context window and enough for most papers and reports in full. Beyond
    # it, chunks are sampled evenly and the answer says so.
    summary_max_chars: int = 120_000

    # ---- Document lifecycle ----------------------------------------------
    # Nothing here is hardcoded elsewhere; the janitor reads only these.

    #: A conversation untouched for this long is archived, and its document
    #: with it if nothing else references the document.
    conversation_inactive_days: int = 90

    #: How long a document sits in `pending_deletion` before it is destroyed.
    #: This window is the only thing standing between an accidental delete and
    #: permanent data loss, so it is deliberately generous.
    document_deletion_grace_days: int = 30

    #: Master switch. Off in tests, so a sweep can never fire mid-assertion.
    maintenance_enabled: bool = True

    #: How often the janitor wakes. Cleanup is measured in days, so hourly is
    #: already far more often than necessary — the cost of a missed run is a
    #: document surviving an extra hour.
    maintenance_interval_minutes: int = 60

    #: Delay before the first sweep, so application startup is never competing
    #: with a purge for connections.
    maintenance_startup_delay_seconds: int = 60

    #: Rows processed per stage per run. Bounds the blast radius of a bad
    #: sweep and keeps one run from monopolising the connection pool.
    maintenance_batch_size: int = 200

    # ---- Validators ------------------------------------------------------

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_cors_origins(cls, value: object) -> object:
        """Accept `a,b,c` from the environment as well as a real list."""
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value

    @field_validator("log_level")
    @classmethod
    def _normalise_log_level(cls, value: str) -> str:
        level = value.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if level not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got {value!r}")
        return level

    # ---- Derived values --------------------------------------------------

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def database_url(self) -> str:
        """Async DSN used by the application (asyncpg driver)."""
        return str(
            PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=self.postgres_user,
                password=self.postgres_password,
                host=self.postgres_host,
                port=self.postgres_port,
                path=self.postgres_db,
            )
        )

    @property
    def test_database_url(self) -> str:
        """Async DSN for the throwaway test database."""
        return str(
            PostgresDsn.build(
                scheme="postgresql+asyncpg",
                username=self.postgres_user,
                password=self.postgres_password,
                host=self.postgres_host,
                port=self.postgres_port,
                path=self.postgres_test_db,
            )
        )

    @property
    def sync_database_url(self) -> str:
        """Sync DSN used by Alembic, which does not run an event loop."""
        return str(
            PostgresDsn.build(
                scheme="postgresql+psycopg",
                username=self.postgres_user,
                password=self.postgres_password,
                host=self.postgres_host,
                port=self.postgres_port,
                path=self.postgres_db,
            )
        )


@lru_cache
def get_settings() -> Settings:
    """Return the singleton settings object.

    Cached so the .env file is parsed once per process, and so tests can
    override configuration with `get_settings.cache_clear()`.
    """
    return Settings()


settings = get_settings()
