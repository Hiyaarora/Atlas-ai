"""A disk-backed embedding cache for the evaluation harness.

WHY
---
The harness's purpose is repeated measurement — A/B a reranker, sweep a chunk
size, re-derive a threshold. Every one of those runs re-embeds the same fixed
corpus and the same fixed 53 queries.

That turned out to be the binding constraint. Gemini's free tier allows 1000
embedding requests per day (`EmbedContentRequestsPerDayPerUserPerProjectPerModel`),
and a day of benchmarking exhausted it mid-A/B, leaving one
configuration measured and the other not.

Embeddings are a pure function of (model, purpose, text), so caching them is
sound rather than a shortcut: the vector for a given query under a given model
is the same today and tomorrow. With this in place the first run of a corpus
costs quota and every subsequent run costs nothing, which is what makes
"measure everything" affordable.

SCOPE
-----
Deliberately in `app/eval`, not `app/embeddings`. Caching query embeddings in
production is a defensible optimisation but a different decision with
different concerns — invalidation, memory, multi-process coherence. This
serves the harness, where the corpus is fixed and the cache can simply be
deleted to force a refresh.

The cache key includes the model name, so switching embedding models does not
silently reuse vectors from a different vector space — the single most
dangerous mistake available here.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from app.core.logging import get_logger
from app.embeddings.base import EmbeddingProvider, EmbeddingPurpose

logger = get_logger(__name__)

DEFAULT_PATH = Path("storage") / "eval_embedding_cache.sqlite"


class CachedEmbeddingProvider(EmbeddingProvider):
    """Wraps another provider, persisting vectors to SQLite."""

    def __init__(self, delegate: EmbeddingProvider, path: Path | None = None) -> None:
        self._delegate = delegate
        self.name = f"cached:{delegate.name}"
        self.model = delegate.model
        self.dimensions = delegate.dimensions

        self._path = path or DEFAULT_PATH
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self._path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS embeddings (key TEXT PRIMARY KEY, vector TEXT NOT NULL)"
        )
        self._connection.commit()

        self.hits = 0
        self.misses = 0

    def _key(self, text: str, purpose: EmbeddingPurpose) -> str:
        # The model is part of the key. Vectors from two different models live
        # in incomparable spaces, and silently mixing them would produce
        # similarity scores that look plausible and mean nothing.
        digest = hashlib.sha256(f"{self.model}|{purpose}|{text}".encode()).hexdigest()
        return digest

    async def embed(self, texts: list[str], *, purpose: EmbeddingPurpose) -> list[list[float]]:
        if not texts:
            return []

        keys = [self._key(text, purpose) for text in texts]
        cached: dict[str, list[float]] = {}

        # One query for the whole batch rather than N round trips.
        placeholders = ",".join("?" for _ in keys)
        rows = self._connection.execute(
            f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})", keys
        ).fetchall()
        for key, vector in rows:
            cached[key] = json.loads(vector)

        missing_indexes = [index for index, key in enumerate(keys) if key not in cached]
        self.hits += len(texts) - len(missing_indexes)
        self.misses += len(missing_indexes)

        if missing_indexes:
            fresh = await self._delegate.embed(
                [texts[index] for index in missing_indexes], purpose=purpose
            )
            for index, vector in zip(missing_indexes, fresh, strict=True):
                cached[keys[index]] = vector
                self._connection.execute(
                    "INSERT OR REPLACE INTO embeddings (key, vector) VALUES (?, ?)",
                    (keys[index], json.dumps(vector)),
                )
            self._connection.commit()

        # Rebuilt in the caller's order. The base contract requires one vector
        # per input in the same order, and a cache that reordered would attach
        # every vector to the wrong chunk.
        return [cached[key] for key in keys]

    def stats(self) -> str:
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total else 0.0
        return f"{self.hits} hits / {self.misses} misses ({rate:.0f}% cached)"
